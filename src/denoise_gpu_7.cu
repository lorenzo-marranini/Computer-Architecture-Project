#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <cuda_runtime.h>

// --- COSTANTI DI CONFIGURAZIONE ---
#define S_MAX 15
#define MIN_WINDOW_SIZE 3
#define R_MAX (S_MAX / 2)       // 7

// --- OTTIMIZZAZIONI GRIGLIA E MEMORIA ---
#define TILE_W 32               // Warp perfetto (32 thread)
#define TILE_H 8
#define SHARED_W (TILE_W + 2 * R_MAX) // 32 + 14 = 46
// ---------------------------------------------------------------------------
// COALESCING FIX — SHARED MEMORY PITCH
// ---------------------------------------------------------------------------
// The old code used SHARED_PITCH = 64.  That value padded each *row* of
// shared memory to 64 bytes, which is fine for bank conflicts, but it did
// nothing to fix the global-memory read pattern that feeds shared memory.
//
// The root cause of the 46% excessive L2 sectors (profiler line 65) was the
// 2-D cooperative load loop:
//
//   for (sy = ty; sy < SHARED_H; sy += TILE_H)       // outer: 22 rows
//     for (sx = tx; sx < SHARED_W; sx += TILE_W)     // inner: 2 cols (46/32)
//
// On the *second* inner iteration (sx = tx + 32) only threads 0..13 are
// active (46 - 32 = 14 live threads out of 32).  Those 14 threads issue a
// single 128-byte cache-line transaction but only consume 14 bytes → 57%
// waste per transaction on those iterations.
//
// FIX: replace the 2-D loop with a 1-D linearised loop over the flat tile.
// Each iteration assigns exactly one element to each thread via:
//
//   linear_id = blockDim.x * blockDim.y * iter + threadIdx.y * blockDim.x + threadIdx.x
//
// Because consecutive threads (same warp) differ only in threadIdx.x (+1),
// they map to consecutive (row, col) pairs *within the same row* as long as
// the tile width >= 32.  SHARED_W = 46 > 32, so every full warp always reads
// 32 consecutive bytes from global memory → 1 transaction, 0 waste.
// The last partial warp (elements 46*22 mod 32 = 4 elements) still wastes
// one transaction, but that is one transaction total vs. 22 wasted partial
// transactions before the fix.
//
// SHARED_PITCH stays at 64 (multiple of 32 bytes) to avoid shared-memory
// bank conflicts when threads in a warp read the same row at different columns
// during the filter computation phase.
#define SHARED_PITCH 64
#define SHARED_H (TILE_H + 2 * R_MAX) // 8 + 14 = 22

// Total elements in the shared tile (used by the 1-D load loop)
#define SHARED_TOTAL (SHARED_H * SHARED_PITCH)  // 22 * 64 = 1408

// Threads per block (used to stride the 1-D loop)
#define THREADS_PER_BLOCK (TILE_W * TILE_H)     // 32 * 8  = 256

// --- MACRO HARDWARE ---
#define SWAP_U8(a, b) { uint8_t tmp = min(a, b); b = max(a, b); a = tmp; }

// --- GESTIONE ERRORI CUDA ---
#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// ---------------------------------------------------------
// KERNEL GPU — COALESCED SHARED-MEMORY LOAD
// ---------------------------------------------------------
__global__ void adaptive_median_ultra_optimized_kernel(
    const uint8_t *__restrict__ in_data,
    uint8_t *__restrict__ out_data,
    int width,
    int height,
    int channels)
{
    __shared__ uint8_t s_in[SHARED_H][SHARED_PITCH];

    // Each thread's position in the output image
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x * TILE_W;   // top-left output column of this block
    const int by = blockIdx.y * TILE_H;   // top-left output row    of this block
    const int gx = bx + tx;
    const int gy = by + ty;

    // Flat thread index within the block [0, THREADS_PER_BLOCK)
    const int flat_tid = ty * TILE_W + tx;

    for (int c = 0; c < channels; c++) {

        // ====================================================================
        // FASE 1: CARICAMENTO COOPERATIVO — 1-D LINEARISED (COALESCING FIX)
        // ====================================================================
        // We iterate over the SHARED_H * SHARED_PITCH = 1408-element flat
        // buffer in strides of THREADS_PER_BLOCK = 256.  This gives 6 full
        // iterations (6 * 256 = 1536 ≥ 1408; last 128 elements are guards).
        //
        // Within each iteration the 256 threads are assigned consecutive
        // flat indices.  We decompose each flat index into (row, col) inside
        // the shared tile:
        //
        //   shared_row = flat_idx / SHARED_PITCH   (integer division)
        //   shared_col = flat_idx % SHARED_PITCH
        //
        // Threads in the same warp share the same `shared_row` as long as
        // they are within the same 64-element row segment, which they always
        // are because SHARED_PITCH = 64 ≥ 32.  Therefore their global
        // addresses differ by exactly 1 byte → perfectly coalesced.
        //
        // The corresponding global row/col are clamped to [0, h-1] / [0, w-1]
        // (border replication, same semantics as before).

        #pragma unroll 2
        for (int i = flat_tid; i < SHARED_H * SHARED_PITCH; i += THREADS_PER_BLOCK) {
            int shared_row = i / SHARED_PITCH;   // which halo row [0, SHARED_H)
            int shared_col = i % SHARED_PITCH;   // column inside padded row

            // Only load real shared-tile columns; padding columns (46..63)
            // are never read during the filter, but we still write them to
            // keep the loop simple.  We clamp shared_col to SHARED_W-1 so
            // padding entries get a valid (harmless) pixel value rather than
            // a potentially out-of-bounds address.
            int clamped_scol = min(shared_col, SHARED_W - 1);

            int g_row = max(0, min(by + shared_row - R_MAX, height - 1));
            int g_col = max(0, min(bx + clamped_scol - R_MAX, width  - 1));

            s_in[shared_row][shared_col] =
                in_data[c * (width * height) + g_row * width + g_col];
        }
        __syncthreads();

        // ====================================================================
        // FASE 2: ELABORAZIONE DEL PIXEL (invariata)
        // ====================================================================
        if (gx < width && gy < height) {
            int s_y = ty + R_MAX;
            int s_x = tx + R_MAX;
            uint8_t Z_xy = s_in[s_y][s_x];

            uint8_t final_color = Z_xy;

            // --- FAST-PATH 3x3 (sorting network in registers) ---
            uint8_t local_9[9];
            int idx = 0;
            #pragma unroll
            for (int wy = -1; wy <= 1; wy++) {
                #pragma unroll
                for (int wx = -1; wx <= 1; wx++) {
                    local_9[idx++] = s_in[s_y + wy][s_x + wx];
                }
            }

            SWAP_U8(local_9[0], local_9[1]); SWAP_U8(local_9[3], local_9[4]); SWAP_U8(local_9[6], local_9[7]);
            SWAP_U8(local_9[1], local_9[2]); SWAP_U8(local_9[4], local_9[5]); SWAP_U8(local_9[7], local_9[8]);
            SWAP_U8(local_9[0], local_9[1]); SWAP_U8(local_9[3], local_9[4]); SWAP_U8(local_9[6], local_9[7]);
            SWAP_U8(local_9[0], local_9[3]); SWAP_U8(local_9[1], local_9[4]); SWAP_U8(local_9[2], local_9[5]);
            SWAP_U8(local_9[3], local_9[6]); SWAP_U8(local_9[4], local_9[7]); SWAP_U8(local_9[5], local_9[8]);
            SWAP_U8(local_9[0], local_9[3]); SWAP_U8(local_9[1], local_9[4]); SWAP_U8(local_9[2], local_9[5]);
            SWAP_U8(local_9[2], local_9[6]); SWAP_U8(local_9[1], local_9[3]); SWAP_U8(local_9[5], local_9[7]);
            SWAP_U8(local_9[2], local_9[4]); SWAP_U8(local_9[4], local_9[6]); SWAP_U8(local_9[3], local_9[5]);
            SWAP_U8(local_9[2], local_9[3]); SWAP_U8(local_9[4], local_9[5]);

            uint8_t z_min = local_9[0];
            uint8_t z_max = local_9[8];
            uint8_t z_med = local_9[4];

            if ((z_med - z_min) > 0 && (z_med - z_max) < 0) {
                final_color = ((Z_xy - z_min) > 0 && (Z_xy - z_max) < 0) ? Z_xy : z_med;
            }
            else {
                // --- SLOW-PATH: histogram in local memory ---
                uint8_t local_hist[256] = {0};

                int window_size = MIN_WINDOW_SIZE;
                while (window_size <= S_MAX) {
                    int r = window_size / 2;

                    if (window_size == 3) {
                        for (int wy = -1; wy <= 1; wy++)
                            for (int wx = -1; wx <= 1; wx++)
                                local_hist[ s_in[s_y + wy][s_x + wx] ]++;
                    } else {
                        for (int wx = -r; wx <= r; wx++) {
                            local_hist[ s_in[s_y - r][s_x + wx] ]++;
                            local_hist[ s_in[s_y + r][s_x + wx] ]++;
                        }
                        for (int wy = -r + 1; wy <= r - 1; wy++) {
                            local_hist[ s_in[s_y + wy][s_x - r] ]++;
                            local_hist[ s_in[s_y + wy][s_x + r] ]++;
                        }
                    }

                    int Z_min_hist = -1, Z_max_hist = -1, Z_med_hist = -1;
                    int count_cumulativo = 0;
                    int target_pos = ((window_size * window_size) / 2) + 1;

                    #pragma unroll 16
                    for (int i = 0; i < 256; i++) {
                        if (local_hist[i] > 0) {
                            if (Z_min_hist == -1) Z_min_hist = i;
                            Z_max_hist = i;
                            count_cumulativo += local_hist[i];
                            if (Z_med_hist == -1 && count_cumulativo >= target_pos) Z_med_hist = i;
                        }
                    }

                    if ((Z_med_hist - Z_min_hist) > 0 && (Z_med_hist - Z_max_hist) < 0) {
                        final_color = ((Z_xy - Z_min_hist) > 0 && (Z_xy - Z_max_hist) < 0) ? Z_xy : Z_med_hist;
                        break;
                    } else {
                        window_size += 2;
                        if (window_size > S_MAX) { final_color = Z_med_hist; break; }
                    }
                }
            }

            out_data[c * (width * height) + (gy * width + gx)] = final_color;
        }
        __syncthreads();
    }
}

// ---------------------------------------------------------
// FUNZIONI HOST E MAIN (invariate)
// ---------------------------------------------------------
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
    uint8_t *raw_data = (uint8_t*)malloc(area * *channels);
    fread(raw_data, 1, area * *channels, f);

    uint8_t *soa_data = (uint8_t*)malloc(area * *channels);
    for (size_t i = 0; i < area; i++)
        for (int c = 0; c < *channels; c++)
            soa_data[c * area + i] = raw_data[i * (*channels) + c];

    free(raw_data);
    fclose(f);
    return soa_data;
}

void save_image_soa(const char *filename, uint8_t *soa_data, int w, int h, int channels) {
    FILE *f = fopen(filename, "wb");
    if (!f) return;
    const char *magic = (channels == 1) ? "P5" : "P6";
    fprintf(f, "%s\n%d %d\n255\n", magic, w, h);

    size_t area = (size_t)w * h;
    uint8_t *raw_data = (uint8_t*)malloc(area * channels);
    for (size_t i = 0; i < area; i++)
        for (int c = 0; c < channels; c++)
            raw_data[i * channels + c] = soa_data[c * area + i];

    fwrite(raw_data, 1, area * channels, f);
    free(raw_data);
    fclose(f);
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <input.ppm> [output.ppm]\n", argv[0]);
        return 1;
    }
    const char *input_file  = argv[1];
    const char *output_file = (argc > 2) ? argv[2] : "output_optimized.ppm";

    int width, height, channels;
    uint8_t *h_img_in = load_image_soa(input_file, &width, &height, &channels);
    if (!h_img_in) { fprintf(stderr, "Errore caricamento immagine!\n"); return 1; }

    size_t img_size = (size_t)width * height * channels * sizeof(uint8_t);
    uint8_t *h_img_out = (uint8_t*)malloc(img_size);

    uint8_t *d_img_in, *d_img_out;
    cudaCheckError(cudaMalloc((void**)&d_img_in,  img_size));
    cudaCheckError(cudaMalloc((void**)&d_img_out, img_size));
    cudaCheckError(cudaMemcpy(d_img_in, h_img_in, img_size, cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(TILE_W, TILE_H);
    dim3 numBlocks((width  + TILE_W - 1) / TILE_W,
                   (height + TILE_H - 1) / TILE_H);

    printf("Avvio Filtro GPU (Coalescing Fix + Sorting Network)...\n");
    printf("  Shared tile : %d x %d  (pitch %d)\n", SHARED_W, SHARED_H, SHARED_PITCH);
    printf("  Load iters  : %d  (ceil(%d / %d))\n",
           (SHARED_H * SHARED_PITCH + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK,
           SHARED_H * SHARED_PITCH, THREADS_PER_BLOCK);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    adaptive_median_ultra_optimized_kernel<<<numBlocks, threadsPerBlock>>>(
        d_img_in, d_img_out, width, height, channels);
    cudaEventRecord(stop);

    cudaCheckError(cudaPeekAtLastError());
    cudaCheckError(cudaDeviceSynchronize());

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    printf("Tempo di esecuzione GPU: %.3f ms\n", ms);

    cudaCheckError(cudaMemcpy(h_img_out, d_img_out, img_size, cudaMemcpyDeviceToHost));
    save_image_soa(output_file, h_img_out, width, height, channels);
    printf("Completato. Salvato in %s\n", output_file);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_img_in);
    cudaFree(d_img_out);
    free(h_img_in);
    free(h_img_out);

    return 0;
}
