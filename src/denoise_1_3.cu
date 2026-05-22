#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <cuda_runtime.h>

// Include Thrust headers for sorting
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>

// ---------------------------------------------------------------------------
// COSTANTI DI CONFIGURAZIONE
// ---------------------------------------------------------------------------
#define S_MAX           15
#define MIN_WINDOW_SIZE 3
#define R_MAX           (S_MAX / 2)     // 7

// Fast-pass tile.
// NOTE: il fast-pass non usa più shared memory, quindi le macro
// SHARED_W / SHARED_PITCH / SHARED_H sono state rimosse.
// TILE_W e TILE_H controllano solo la forma del blocco (32x8 = 256 thread,
// warp allineati lungo x → letture coalescenti su global memory).
#define TILE_W          32
#define TILE_H          8
#define THREADS_PER_BLOCK (TILE_W * TILE_H)    // 256

// ---------------------------------------------------------------------------
// SLOW-PASS TILE
// ---------------------------------------------------------------------------
// SLOW_BLOCK_SIZE overridabile via -DBLOCK_Y per il sweep.
// BLOCK_Y rappresenta i warp per blocco nel slow_pass kernel.
// fast_pass resta invariato.
//
// Anche __launch_bounds__ viene scalato proporzionalmente per mantenere
// l'occupancy target di 2048 thread/SM (= max teorico T4 / 2):
//   SLOW_BLOCK_SIZE=128 → (128, 16)
//   SLOW_BLOCK_SIZE=256 → (256, 8)
//   SLOW_BLOCK_SIZE=512 → (512, 4)  [originale]
//   SLOW_BLOCK_SIZE=1024 → (1024, 2)
#ifdef BLOCK_Y
#define SLOW_BLOCK_SIZE (32 * BLOCK_Y)
#else
#define SLOW_BLOCK_SIZE 512
#endif

// Target di blocchi/SM per launch_bounds: 2048/N, minimo 1
#define SLOW_BLOCKS_PER_SM ((2048 / SLOW_BLOCK_SIZE) > 0 ? (2048 / SLOW_BLOCK_SIZE) : 1)

// ---------------------------------------------------------------------------
// MACRO HARDWARE
// ---------------------------------------------------------------------------
#define SWAP_U8(a, b) { uint8_t tmp = min(a,b); b = max(a,b); a = tmp; }

// ---------------------------------------------------------------------------
// GESTIONE ERRORI CUDA
// ---------------------------------------------------------------------------
#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true)
{
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n",
                cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// ---------------------------------------------------------------------------
// Helper device: legge in_data[c, y, x] con clamp ai bordi.
// SoA: piano del canale c a offset c * area.
// ---------------------------------------------------------------------------
__device__ __forceinline__ uint8_t load_clamped(
    const uint8_t *__restrict__ in_data,
    int c, int y, int x, int width, int height, int area)
{
    int cy = max(0, min(y, height - 1));
    int cx = max(0, min(x, width  - 1));
    return in_data[c * area + cy * width + cx];
}

// ===========================================================================
// KERNEL 1 — FAST PASS  (GLOBAL MEMORY, no shared)
// ===========================================================================
// Differenza rispetto alla versione shared:
//   - Niente __shared__ s_in[][], niente caricamento cooperativo, niente
//     __syncthreads() in questo kernel.
//   - I 9 pixel della finestra 3×3 vengono letti direttamente da in_data.
//   - Per i pixel interni (is_safe) si usa pointer arithmetic da un base
//     precalcolato → niente clamp, accessi lineari coalescenti.
//   - Per i pixel di bordo si usa load_clamped (max/min).
// La compaction (atomicAdd su slow_count + scrittura in slow_list) è
// identica a prima.
// ===========================================================================
__global__ void fast_pass_kernel(
    const uint8_t *__restrict__ in_data,
    uint8_t       *__restrict__ out_data,
    int width, int height, int channels,
    int  *__restrict__ slow_list,
    int  *__restrict__ slow_count,
    int  area)
{
    const int gx = blockIdx.x * TILE_W + threadIdx.x;
    const int gy = blockIdx.y * TILE_H + threadIdx.y;

    if (gx >= width || gy >= height) return;

    // Pixel interno: tutta la finestra 3×3 sta dentro l'immagine?
    const bool is_safe = (gx >= 1 && gx < width  - 1 &&
                          gy >= 1 && gy < height - 1);

    for (int c = 0; c < channels; c++) {

        const int pixel_idx = gy * width + gx;
        const int base      = c * area + pixel_idx;
        const uint8_t Z_xy  = in_data[base];

        // ------------------------------------------------------------------
        // Carico i 9 pixel del 3×3 direttamente da global memory.
        // ------------------------------------------------------------------
        uint8_t w9[9];

        if (is_safe) {
            // Accessi lineari: thread con threadIdx.x consecutivi leggono
            // indirizzi consecutivi → letture coalescenti.
            w9[0] = in_data[base - width - 1];
            w9[1] = in_data[base - width    ];
            w9[2] = in_data[base - width + 1];
            w9[3] = in_data[base         - 1];
            w9[4] = in_data[base            ];
            w9[5] = in_data[base         + 1];
            w9[6] = in_data[base + width - 1];
            w9[7] = in_data[base + width    ];
            w9[8] = in_data[base + width + 1];
        } else {
            int idx = 0;
            #pragma unroll
            for (int wy = -1; wy <= 1; wy++) {
                #pragma unroll
                for (int wx = -1; wx <= 1; wx++) {
                    w9[idx++] = load_clamped(in_data, c,
                                             gy + wy, gx + wx,
                                             width, height, area);
                }
            }
        }

        // ------------------------------------------------------------------
        // Sorting network 9 elementi (identica alla versione shared)
        // ------------------------------------------------------------------
        SWAP_U8(w9[0],w9[1]); SWAP_U8(w9[3],w9[4]); SWAP_U8(w9[6],w9[7]);
        SWAP_U8(w9[1],w9[2]); SWAP_U8(w9[4],w9[5]); SWAP_U8(w9[7],w9[8]);
        SWAP_U8(w9[0],w9[1]); SWAP_U8(w9[3],w9[4]); SWAP_U8(w9[6],w9[7]);
        SWAP_U8(w9[0],w9[3]); SWAP_U8(w9[1],w9[4]); SWAP_U8(w9[2],w9[5]);
        SWAP_U8(w9[3],w9[6]); SWAP_U8(w9[4],w9[7]); SWAP_U8(w9[5],w9[8]);
        SWAP_U8(w9[0],w9[3]); SWAP_U8(w9[1],w9[4]); SWAP_U8(w9[2],w9[5]);
        SWAP_U8(w9[2],w9[6]); SWAP_U8(w9[1],w9[3]); SWAP_U8(w9[5],w9[7]);
        SWAP_U8(w9[2],w9[4]); SWAP_U8(w9[4],w9[6]); SWAP_U8(w9[3],w9[5]);
        SWAP_U8(w9[2],w9[3]); SWAP_U8(w9[4],w9[5]);

        const uint8_t z_min = w9[0];
        const uint8_t z_max = w9[8];
        const uint8_t z_med = w9[4];

        if ((z_med - z_min) > 0 && (z_med - z_max) < 0) {
            // Fast path: scrivo il pixel finale.
            out_data[base] =
                ((Z_xy - z_min) > 0 && (Z_xy - z_max) < 0) ? Z_xy : z_med;
        } else {
            // Slow path needed: placeholder + compaction.
            out_data[base] = Z_xy;
            int slot = atomicAdd(&slow_count[c], 1);
            slow_list[c * area + slot] = pixel_idx;
        }
    }
}

// ===========================================================================
// KERNEL 2 — SLOW PASS  (CORRECT: per-thread local-memory histogram)
// ===========================================================================
// BUGFIX: the previous version declared the histogram as
//   __shared__ uint8_t s_hist[WARPS_PER_BLOCK][256];
//   uint8_t* my_hist = s_hist[threadIdx.x >> 5];
// so all 32 threads of a warp shared ONE 256-byte histogram.  Since each
// thread processes a DIFFERENT pixel, their counts collided into the same
// slice and every thread read a meaningless merged median — leaving the
// output image still noisy.
//
// A correct per-thread histogram in shared memory would need
// SLOW_BLOCK_SIZE * 256 bytes (e.g. 512 * 256 = 128 KB), far exceeding the
// 48-96 KB shared-memory limit.  Therefore the histogram MUST live in
// per-thread local memory (registers where possible, L1-backed local mem
// otherwise) — which is exactly what the working two-pass version uses.
//
// Optimizations kept (both correct and independent of the histogram bug):
//   - Incremental ring update (fix #5): first iteration loads the full 3×3,
//     later iterations add only the new outer ring.
//   - Two-pass early-exit scan (fix #3): forward scan exits when the median
//     bin is reached; backward scan exits on the first non-zero bin (max).
// ===========================================================================
__global__ void slow_pass_kernel(
    const uint8_t *__restrict__ in_data,
    uint8_t       *__restrict__ out_data,
    int width, int height, int channels,
    const int *__restrict__ slow_list,
    const int *__restrict__ slow_count,
    int area)
{
    // One thread per impulse pixel; 1-D grid.
    for (int c = 0; c < channels; c++) {

        const int n   = slow_count[c];
        const int tid = blockIdx.x * SLOW_BLOCK_SIZE + threadIdx.x;

        if (tid >= n) continue;          // excess threads do nothing

        const int pixel_idx = slow_list[c * area + tid];
        const int gx = pixel_idx % width;
        const int gy = pixel_idx / width;
        const uint8_t Z_xy = in_data[c * area + pixel_idx];

        // Per-thread histogram in LOCAL memory (NOT shared).
        // Each thread owns its own 256-byte histogram → no collision.
        uint8_t local_hist[256] = {0};

        int window_size = MIN_WINDOW_SIZE;
        uint8_t final_color = Z_xy;

        // ── Incremental fill: full 3×3 on the first iteration ──────────────
        for (int wy = -1; wy <= 1; wy++) {
            int gr = max(0, min(gy + wy, height - 1));
            for (int wx = -1; wx <= 1; wx++) {
                int gc = max(0, min(gx + wx, width - 1));
                local_hist[ in_data[c * area + gr * width + gc] ]++;
            }
        }

        bool decided = false;

        while (!decided && window_size <= S_MAX) {

            // Add only the new outer ring for window_size > 3
            if (window_size > 3) {
                const int r = window_size / 2;
                for (int wx = -r; wx <= r; wx++) {
                    int gr_top = max(0, min(gy - r, height - 1));
                    int gr_bot = max(0, min(gy + r, height - 1));
                    int gc     = max(0, min(gx + wx, width  - 1));
                    local_hist[ in_data[c * area + gr_top * width + gc] ]++;
                    local_hist[ in_data[c * area + gr_bot * width + gc] ]++;
                }
                for (int wy = -r + 1; wy <= r - 1; wy++) {
                    int gr     = max(0, min(gy + wy, height - 1));
                    int gc_lft = max(0, min(gx - r,  width  - 1));
                    int gc_rgt = max(0, min(gx + r,  width  - 1));
                    local_hist[ in_data[c * area + gr * width + gc_lft] ]++;
                    local_hist[ in_data[c * area + gr * width + gc_rgt] ]++;
                }
            }

            // ── Two-pass early-exit scan (fix #3) ──────────────────────────
            int Z_min_h = -1, Z_med_h = -1, Z_max_h = -1;
            int cum = 0;
            const int target = ((window_size * window_size) / 2) + 1;

            for (int i = 0; i < 256; i++) {
                int count = local_hist[i];
                if (count > 0) {
                    if (Z_min_h == -1) Z_min_h = i;
                    cum += count;
                    if (cum >= target) { Z_med_h = i; break; }
                }
            }
            for (int i = 255; i >= 0; i--) {
                if (local_hist[i] > 0) { Z_max_h = i; break; }
            }

            // ── Adaptive median decision ───────────────────────────────────
            if ((Z_med_h - Z_min_h) > 0 && (Z_med_h - Z_max_h) < 0) {
                final_color = ((Z_xy - Z_min_h) > 0 && (Z_xy - Z_max_h) < 0)
                              ? Z_xy : (uint8_t)Z_med_h;
                decided = true;
            } else {
                window_size += 2;
                if (window_size > S_MAX) {
                    final_color = (Z_med_h >= 0) ? (uint8_t)Z_med_h : Z_xy;
                    decided = true;
                }
            }
        }

        out_data[c * area + pixel_idx] = final_color;
    }
}

// ===========================================================================
// FUNZIONI HOST
// ===========================================================================
uint8_t* load_image_soa(const char *filename, int *w, int *h, int *channels) {
    FILE *f = fopen(filename, "rb");
    if (!f) return NULL;
    char magic[3];
    fscanf(f, "%2s", magic);
    if      (strcmp(magic, "P5") == 0) *channels = 1;
    else if (strcmp(magic, "P6") == 0) *channels = 3;
    else { fclose(f); return NULL; }
    int max_val;
    fscanf(f, "%d %d %d", w, h, &max_val);
    fgetc(f);
    size_t area = (size_t)(*w) * (*h);
    uint8_t *raw  = (uint8_t*)malloc(area * *channels);
    uint8_t *soa  = (uint8_t*)malloc(area * *channels);
    fread(raw, 1, area * *channels, f);
    for (size_t i = 0; i < area; i++)
        for (int c = 0; c < *channels; c++)
            soa[c * area + i] = raw[i * (*channels) + c];
    free(raw);
    fclose(f);
    return soa;
}

void save_image_soa(const char *filename, uint8_t *soa, int w, int h, int channels) {
    FILE *f = fopen(filename, "wb");
    if (!f) return;
    fprintf(f, "%s\n%d %d\n255\n", (channels == 1) ? "P5" : "P6", w, h);
    size_t area = (size_t)w * h;
    uint8_t *raw = (uint8_t*)malloc(area * channels);
    for (size_t i = 0; i < area; i++)
        for (int c = 0; c < channels; c++)
            raw[i * channels + c] = soa[c * area + i];
    fwrite(raw, 1, area * channels, f);
    free(raw);
    fclose(f);
}

// ===========================================================================
// MAIN
// ===========================================================================
int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <input.ppm> [output.ppm]\n", argv[0]);
        return 1;
    }
    const char *input_file  = argv[1];
    const char *output_file = (argc > 2) ? argv[2] : "output_twopass.ppm";

    int width, height, channels;
    uint8_t *h_in = load_image_soa(input_file, &width, &height, &channels);
    if (!h_in) { fprintf(stderr, "Errore caricamento immagine!\n"); return 1; }

    const int   area     = width * height;
    const size_t img_sz  = (size_t)area * channels;

    uint8_t *d_in, *d_out;
    cudaCheckError(cudaMalloc(&d_in,  img_sz));
    cudaCheckError(cudaMalloc(&d_out, img_sz));
    cudaCheckError(cudaMemcpy(d_in, h_in, img_sz, cudaMemcpyHostToDevice));

    int *d_slow_list, *d_slow_count;
    cudaCheckError(cudaMalloc(&d_slow_list,  (size_t)area * channels * sizeof(int)));
    cudaCheckError(cudaMalloc(&d_slow_count, channels * sizeof(int)));
    cudaCheckError(cudaMemset(d_slow_count, 0, channels * sizeof(int)));

    dim3 fast_block(TILE_W, TILE_H);
    dim3 fast_grid((width  + TILE_W - 1) / TILE_W,
                   (height + TILE_H - 1) / TILE_H);

    printf("=== Two-Pass Adaptive Median Filter (fast pass: global memory) ===\n");
    printf("Image : %d x %d x %d ch\n", width, height, channels);

    cudaEvent_t t0, t1, t2;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    cudaEventCreate(&t2);

    // ── PASS 1 ──────────────────────────────────────────────────────────────
    cudaEventRecord(t0);
    fast_pass_kernel<<<fast_grid, fast_block>>>(
        d_in, d_out, width, height, channels,
        d_slow_list, d_slow_count, area);
    cudaCheckError(cudaPeekAtLastError());
    cudaEventRecord(t1);
    cudaCheckError(cudaDeviceSynchronize());

    int h_slow_count[3] = {0, 0, 0};
    cudaCheckError(cudaMemcpy(h_slow_count, d_slow_count,
                              channels * sizeof(int), cudaMemcpyDeviceToHost));

    float ms_fast = 0;
    cudaEventElapsedTime(&ms_fast, t0, t1);
    printf("Pass 1 (fast) : %.3f ms\n", ms_fast);

    for (int c = 0; c < channels; c++)
        printf("  Ch %d: %d / %d pixels need slow path (%.1f%%)\n",
               c, h_slow_count[c], area,
               100.0f * h_slow_count[c] / area);

    // ── SORTING SLOW_LIST (The L2 Cache / Memory Coalescing Fix) ────────────
    for (int c = 0; c < channels; c++) {
        if (h_slow_count[c] > 0) {
            // Sort the indices for this channel so neighboring threads in 
            // the slow_pass warp process spatially adjacent pixels.
            thrust::device_ptr<int> d_ptr(d_slow_list + c * area);
            thrust::sort(thrust::device, d_ptr, d_ptr + h_slow_count[c]);
        }
    }

    // ── PASS 2 ──────────────────────────────────────────────────────────────
    int max_slow = 0;
    for (int c = 0; c < channels; c++)
        if (h_slow_count[c] > max_slow) max_slow = h_slow_count[c];

    if (max_slow > 0) {
        int slow_grid = (max_slow + SLOW_BLOCK_SIZE - 1) / SLOW_BLOCK_SIZE;
        
        // Re-record t1 to isolate slow-pass execution time accurately
        cudaEventRecord(t1);
        
        slow_pass_kernel<<<slow_grid, SLOW_BLOCK_SIZE>>>(
            d_in, d_out, width, height, channels,
            d_slow_list, d_slow_count, area);
            
        cudaCheckError(cudaPeekAtLastError());
        cudaEventRecord(t2);
        cudaCheckError(cudaDeviceSynchronize());

        float ms_slow = 0;
        cudaEventElapsedTime(&ms_slow, t1, t2);
        printf("Pass 2 (slow) : %.3f ms  (%d threads)\n", ms_slow, max_slow);
    } else {
        printf("Pass 2 (slow) : skipped (no impulse pixels)\n");
        cudaEventRecord(t2);
    }

    float ms_total = 0;
    cudaEventElapsedTime(&ms_total, t0, t2);
    printf("Total GPU time: %.3f ms (includes sorting)\n", ms_total);

    // Copy result back and save
    uint8_t *h_out = (uint8_t*)malloc(img_sz);
    cudaCheckError(cudaMemcpy(h_out, d_out, img_sz, cudaMemcpyDeviceToHost));
    save_image_soa(output_file, h_out, width, height, channels);
    printf("Salvato in %s\n", output_file);

    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    cudaEventDestroy(t2);
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_slow_list);
    cudaFree(d_slow_count);
    free(h_in);
    free(h_out);

    return 0;
}

