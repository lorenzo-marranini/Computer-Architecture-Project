#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// COSTANTI DI CONFIGURAZIONE
// ---------------------------------------------------------------------------
#define S_MAX           15
#define MIN_WINDOW_SIZE 3
#define R_MAX           (S_MAX / 2)     // 7

// Fast-pass tile (same as before, keeps occupancy high)
#define TILE_W          32
#define TILE_H          8
#define SHARED_W        (TILE_W + 2 * R_MAX)   // 46
#define SHARED_PITCH    64                      // ≥ SHARED_W, multiple of 32
#define SHARED_H        (TILE_H + 2 * R_MAX)   // 22
#define THREADS_PER_BLOCK (TILE_W * TILE_H)    // 256

// ---------------------------------------------------------------------------
// SLOW-PASS TILE
// ---------------------------------------------------------------------------
// Each thread in the slow-pass kernel is a confirmed impulse pixel.
// We use a flat 1-D launch (512 threads/block) — no 2-D tile needed because
// every thread accesses a different arbitrary (gx, gy) location; coalescing
// of global reads is impossible here anyway, so simplicity wins.
// The slow-pass thread reloads its own (S_MAX×S_MAX) = 15×15 halo directly
// from global memory (L2-cached); no shared memory is used, which avoids
// the __syncthreads() barrier that would be meaningless in an irregular kernel.
#define SLOW_BLOCK_SIZE 512

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

// ===========================================================================
// KERNEL 1 — FAST PASS
// ===========================================================================
// Every thread runs the 3×3 sorting-network test.
//   • Fast pixels  → result written to out_data, mask byte = 0.
//   • Impulse pixels → input pixel copied to out_data (placeholder),
//                      packed coordinate appended to slow_list[] via atomic,
//                      mask byte = 1 (unused after compaction but useful for
//                      debugging / future multi-channel loops).
//
// DIVERGENCE situation AFTER this fix:
//   The if/else on the fast-path test still exists, but its two branches are
//   now both trivial (one store vs. one store + one atomicAdd).  The expensive
//   while-loop histogram is gone from this kernel entirely → warp divergence
//   cost drops from ~hundreds of cycles to ~2-3 cycles.
// ===========================================================================
__global__ void fast_pass_kernel(
    const uint8_t *__restrict__ in_data,
    uint8_t       *__restrict__ out_data,
    int width, int height, int channels,
    // Compaction outputs (one list per channel, pre-allocated to worst-case)
    int  *__restrict__ slow_list,     // packed pixel indices: [channel * area + idx]
    int  *__restrict__ slow_count,    // one atomic counter per channel
    int  area)                        // width * height, passed to avoid remul
{
    __shared__ uint8_t s_in[SHARED_H][SHARED_PITCH];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x * TILE_W;
    const int by = blockIdx.y * TILE_H;
    const int gx = bx + tx;
    const int gy = by + ty;
    const int flat_tid = ty * TILE_W + tx;

    for (int c = 0; c < channels; c++) {

        // ------------------------------------------------------------------
        // FASE 1: CARICAMENTO COOPERATIVO COALESCENTE (1-D linearised)
        // ------------------------------------------------------------------
        #pragma unroll 2
        for (int i = flat_tid; i < SHARED_H * SHARED_PITCH; i += THREADS_PER_BLOCK) {
            int sr  = i / SHARED_PITCH;
            int sc  = i % SHARED_PITCH;
            int csc = min(sc, SHARED_W - 1);
            int gr  = max(0, min(by + sr - R_MAX, height - 1));
            int gc  = max(0, min(bx + csc - R_MAX, width  - 1));
            s_in[sr][sc] = in_data[c * area + gr * width + gc];
        }
        __syncthreads();

        // ------------------------------------------------------------------
        // FASE 2: TEST 3×3 — NO SLOW PATH HERE
        // ------------------------------------------------------------------
        if (gx < width && gy < height) {
            const int s_y = ty + R_MAX;
            const int s_x = tx + R_MAX;
            const uint8_t Z_xy = s_in[s_y][s_x];

            // Load 3×3 window into registers
            uint8_t w9[9];
            int idx = 0;
            #pragma unroll
            for (int wy = -1; wy <= 1; wy++)
                #pragma unroll
                for (int wx = -1; wx <= 1; wx++)
                    w9[idx++] = s_in[s_y + wy][s_x + wx];

            // Sorting network — 9-element, branchless
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

            const int pixel_idx = gy * width + gx;

            if ((z_med - z_min) > 0 && (z_med - z_max) < 0) {
                // ── FAST PATH: median is valid ──────────────────────────────
                // Write final pixel immediately; no slow-path entry needed.
                out_data[c * area + pixel_idx] =
                    ((Z_xy - z_min) > 0 && (Z_xy - z_max) < 0) ? Z_xy : z_med;
            } else {
                // ── SLOW PATH NEEDED ────────────────────────────────────────
                // Write a placeholder (original pixel) so the output buffer
                // is always fully populated.  The slow_pass_kernel will
                // overwrite this slot with the correct value.
                out_data[c * area + pixel_idx] = Z_xy;

                // Append this pixel's flat index to the per-channel slow list.
                // atomicAdd returns the old counter value = our slot index.
                // ---------------------------------------------------------------------------
                // WHY atomicAdd IS SAFE HERE:
                //   • slow_count[c] starts at 0 (zeroed by cudaMemset on host).
                //   • In the absolute worst case every pixel is impulse noise
                //     → slow_count[c] reaches area = width*height, which is
                //     exactly the allocated size of slow_list[c*area .. (c+1)*area-1].
                //   • atomicAdd on a 32-bit global int is a single hardware
                //     instruction on all CUDA devices; no race condition possible.
                // ---------------------------------------------------------------------------
                int slot = atomicAdd(&slow_count[c], 1);
                slow_list[c * area + slot] = pixel_idx;
            }
        }
        __syncthreads();
    }
}

// ===========================================================================
// KERNEL 2 — SLOW PASS (divergence-free)
// ===========================================================================
// One thread per impulse pixel.  Every thread in this kernel is known to
// need the histogram loop → zero divergence on that branch.
//
// The remaining minor divergence source is the `while (window_size <= S_MAX)`
// loop itself: different pixels may converge at different window sizes.
// This is unavoidable without a fundamentally different algorithm, but its
// impact is now isolated to a small minority of pixels (typical impulse noise
// is < 30% of pixels for realistic noise levels).
//
// Memory access pattern:
//   Each thread reads up to a 15×15 = 225-pixel halo from in_data.
//   These are scattered reads (different (gx,gy) per thread) and will
//   hit L2 cache since fast_pass already warmed it.
// ===========================================================================
__global__ void slow_pass_kernel(
    const uint8_t *__restrict__ in_data,
    uint8_t       *__restrict__ out_data,
    int width, int height, int channels,
    const int *__restrict__ slow_list,
    const int *__restrict__ slow_count,
    int area)
{
    // This kernel is launched with a 1-D grid, one thread per slow pixel.
    // blockIdx.x and threadIdx.x together address the slow_list for channel c.
    // We loop over channels in the outer loop so we can reuse the same grid.

    for (int c = 0; c < channels; c++) {

        const int n = slow_count[c];   // number of impulse pixels this channel
        const int tid = blockIdx.x * SLOW_BLOCK_SIZE + threadIdx.x;

        if (tid >= n) continue;        // excess threads do nothing

        const int pixel_idx = slow_list[c * area + tid];
        const int gx = pixel_idx % width;
        const int gy = pixel_idx / width;
        const uint8_t Z_xy = in_data[c * area + pixel_idx];

        // Per-thread histogram — lives in local memory (registers where
        // possible, spills to L1-backed local mem otherwise).
        // This is the same as before; improvement #4 (shared histogram)
        // is a separate optimisation on top of this one.
        uint8_t local_hist[256] = {0};

        int window_size = MIN_WINDOW_SIZE;
        uint8_t final_color = Z_xy;   // safe default

        while (window_size <= S_MAX) {
            const int r = window_size / 2;

            // Incrementally build histogram: on first iteration load full 3×3,
            // on subsequent iterations add only the new border ring.
            if (window_size == 3) {
                for (int wy = -1; wy <= 1; wy++)
                    for (int wx = -1; wx <= 1; wx++) {
                        int gr = max(0, min(gy + wy, height - 1));
                        int gc = max(0, min(gx + wx, width  - 1));
                        local_hist[ in_data[c * area + gr * width + gc] ]++;
                    }
            } else {
                // Add only the outermost ring of the new window size
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

            // Scan histogram for min / median / max
            int Z_min_h = -1, Z_max_h = -1, Z_med_h = -1;
            int cum = 0;
            const int target = ((window_size * window_size) / 2) + 1;

            #pragma unroll 16
            for (int i = 0; i < 256; i++) {
                if (local_hist[i] > 0) {
                    if (Z_min_h == -1) Z_min_h = i;
                    Z_max_h = i;
                    cum += local_hist[i];
                    if (Z_med_h == -1 && cum >= target) Z_med_h = i;
                }
            }

            if ((Z_med_h - Z_min_h) > 0 && (Z_med_h - Z_max_h) < 0) {
                final_color = ((Z_xy - Z_min_h) > 0 && (Z_xy - Z_max_h) < 0)
                              ? Z_xy : (uint8_t)Z_med_h;
                break;
            } else {
                window_size += 2;
                if (window_size > S_MAX) {
                    final_color = (Z_med_h >= 0) ? (uint8_t)Z_med_h : Z_xy;
                    break;
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

    // Device image buffers
    uint8_t *d_in, *d_out;
    cudaCheckError(cudaMalloc(&d_in,  img_sz));
    cudaCheckError(cudaMalloc(&d_out, img_sz));
    cudaCheckError(cudaMemcpy(d_in, h_in, img_sz, cudaMemcpyHostToDevice));

    // -----------------------------------------------------------------------
    // COMPACTION BUFFERS
    // -----------------------------------------------------------------------
    // slow_list : worst-case every pixel is impulse → area entries per channel
    // slow_count: one int per channel, atomically incremented by fast_pass
    // -----------------------------------------------------------------------
    int *d_slow_list, *d_slow_count;
    cudaCheckError(cudaMalloc(&d_slow_list,  (size_t)area * channels * sizeof(int)));
    cudaCheckError(cudaMalloc(&d_slow_count, channels * sizeof(int)));
    cudaCheckError(cudaMemset(d_slow_count, 0, channels * sizeof(int)));

    // Grid for fast_pass (same tiling as before)
    dim3 fast_block(TILE_W, TILE_H);
    dim3 fast_grid((width  + TILE_W - 1) / TILE_W,
                   (height + TILE_H - 1) / TILE_H);

    printf("=== Two-Pass Adaptive Median Filter ===\n");
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

    // Read back how many impulse pixels were found (cheap: channels ints)
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

    // ── PASS 2 ──────────────────────────────────────────────────────────────
    // Grid size = ceil(max_slow_count / SLOW_BLOCK_SIZE).
    // We use the max across channels to keep a single grid dimension; threads
    // for channels with fewer impulse pixels exit early via the `if (tid >= n)`
    // guard inside the kernel.
    int max_slow = 0;
    for (int c = 0; c < channels; c++)
        if (h_slow_count[c] > max_slow) max_slow = h_slow_count[c];

    if (max_slow > 0) {
        int slow_grid = (max_slow + SLOW_BLOCK_SIZE - 1) / SLOW_BLOCK_SIZE;
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
    printf("Total GPU time: %.3f ms\n", ms_total);

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
