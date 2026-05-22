#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <limits.h>
#include <cuda_runtime.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>


//versione errata
// ===========================================================================
// CONFIGURAZIONE
// ===========================================================================
#define S_MAX           15
#define MIN_WINDOW_SIZE 3
#define R_MAX           (S_MAX / 2)     // 7

#define TILE_W          32
#define TILE_H          8
#define SHARED_PITCH    64
#define SHARED_W        (TILE_W + 2 * R_MAX)   // 46
#define SHARED_H        (TILE_H + 2 * R_MAX)   // 22
#define THREADS_PER_BLOCK (TILE_W * TILE_H)    // 256

// SLOW_BLOCK_SIZE overridabile via -DBLOCK_Y per il sweep.
// BLOCK_Y rappresenta i warp per blocco nel slow_pass kernel.
// fast_pass resta invariato.
#ifdef BLOCK_Y
#define SLOW_BLOCK_SIZE (32 * BLOCK_Y)
#else
#define SLOW_BLOCK_SIZE 512
#endif

// Target di blocchi/SM per launch_bounds: 2048/N, minimo 1
#define SLOW_BLOCKS_PER_SM ((2048 / SLOW_BLOCK_SIZE) > 0 ? (2048 / SLOW_BLOCK_SIZE) : 1)
#define WARPS_PER_BLOCK (SLOW_BLOCK_SIZE / 32) // 16

// ---------------------------------------------------------------------------
// SLOW-PASS COOPERATIVE TILE (Fix #1)
// ---------------------------------------------------------------------------
// After thrust::sort, the 512 pixels of a block are spatially clustered:
// typically they span 2-3 contiguous image rows.  We cooperatively load a
// bounding-box tile of in_data into shared memory, then every thread reads
// its 15x15 halo from shared memory instead of issuing 225 scattered
// __ldg calls.
//
// The bounding box is computed at runtime by a block-wide min/max reduction
// over the (gx, gy) of the block's pixels.  We cap the tile size at
// SLOW_TILE_MAX_W * SLOW_TILE_MAX_H bytes; if the bounding box exceeds the
// cap (rare, for the last block of a channel), we fall back to global reads.
// ---------------------------------------------------------------------------
#define SLOW_TILE_MAX_W 192   // typical sorted-block width + 2*R_MAX
#define SLOW_TILE_MAX_H 32    // typical sorted-block height + 2*R_MAX
// Total shared mem for the tile: 192 * 32 = 6144 bytes
// Plus the histogram: 16 * 256 = 4096 bytes
// Total per block: ~10 KB, well within 32 KB available

#define SWAP_U8(a, b) { uint8_t tmp = min(a,b); b = max(a,b); a = tmp; }

#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// ===========================================================================
// KERNEL 1 — FAST PASS (unchanged from previous version)
// ===========================================================================
__global__ void fast_pass_kernel(
    const uint8_t* __restrict__ in_data,
    uint8_t* __restrict__ out_data,
    int width, int height, int channels,
    int* __restrict__ slow_list, int* __restrict__ slow_count, int area)
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

        #pragma unroll 2
        for (int i = flat_tid; i < SHARED_H * SHARED_PITCH; i += THREADS_PER_BLOCK) {
            int sr  = i / SHARED_PITCH;
            int sc  = i % SHARED_PITCH;
            int csc = min(sc, SHARED_W - 1);
            int gr  = max(0, min(by + sr - R_MAX, height - 1));
            int gc  = max(0, min(bx + csc - R_MAX, width  - 1));
            s_in[sr][sc] = in_data[c * area + gr * width + gc];
        }
        __syncthreads();

        if (gx < width && gy < height) {
            const int s_y = ty + R_MAX;
            const int s_x = tx + R_MAX;
            const uint8_t Z_xy = s_in[s_y][s_x];

            uint8_t w9[9];
            int idx = 0;
            #pragma unroll
            for (int wy = -1; wy <= 1; wy++)
                #pragma unroll
                for (int wx = -1; wx <= 1; wx++)
                    w9[idx++] = s_in[s_y + wy][s_x + wx];

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
                out_data[c * area + pixel_idx] =
                    ((Z_xy - z_min) > 0 && (Z_xy - z_max) < 0) ? Z_xy : z_med;
            } else {
                out_data[c * area + pixel_idx] = Z_xy;
                int slot = atomicAdd(&slow_count[c], 1);
                slow_list[c * area + slot] = pixel_idx;
            }
        }
        __syncthreads();
    }
}

// ===========================================================================
// KERNEL 2 — SLOW PASS with FIXES #5, #3, #1
// ===========================================================================
//
// FIX #5 — INCREMENTAL HISTOGRAM UPDATES
// ───────────────────────────────────────
// The previous version rebuilt the entire (2r+1)² histogram on every
// iteration of the adaptive window loop.  Worst case: 9+25+49+81+121+169+225
// = 679 increments to reach window=15.
//
// The fix: on the first iteration build the full 3×3 (9 increments), then
// on each subsequent iteration add ONLY the new outer ring (8r-4 elements
// for window 2r+1).  Total: 9+12+20+28+36+44+52 = 201 increments — over
// 3× fewer histogram operations.
//
// FIX #3 — EARLY-EXIT HISTOGRAM SCAN
// ──────────────────────────────────
// The previous scan iterated all 256 bins even after finding the median.
// New approach: two short scans.
//   • Forward scan: accumulate count, exit as soon as cumulative count ≥
//     target (median found).  Z_min_h is the first non-zero bin seen.
//   • Backward scan from 255: exit on first non-zero bin (max found).
// For natural images the median is typically in bins ~50-150, so the
// forward scan exits after ~50-150 iterations instead of always 256.
// Backward scan typically exits within 10-50 iterations.
// Combined: ~50% fewer scan operations on average.
//
// FIX #1 — COOPERATIVE TILE LOADING
// ─────────────────────────────────
// After thrust::sort, the block's 512 pixels are spatially clustered.
// We compute the (min_gx, min_gy, max_gx, max_gy) bounding box of the
// block's pixels using a block-wide reduction in shared memory, then
// cooperatively load the bounding box + 7-pixel halo into shared memory.
// Each thread then reads its 15×15 window from shared memory at ~4-cycle
// latency instead of from L2 at ~200-cycle latency with 81% uncoalesced
// excess.
//
// FALLBACK: if the bounding box exceeds SLOW_TILE_MAX_W × SLOW_TILE_MAX_H
// (rare — happens at channel boundaries or edges), the block falls back
// to direct global reads via __ldg.
// ===========================================================================

__global__ __launch_bounds__(SLOW_BLOCK_SIZE, SLOW_BLOCKS_PER_SM)
void slow_pass_kernel(
    const uint8_t* __restrict__ in_data,
    uint8_t* __restrict__ out_data,
    int width, int height, int channels,
    const int* __restrict__ slow_list, const int* __restrict__ slow_count, int area)
{
    // ── SHARED MEMORY LAYOUT ───────────────────────────────────────────────
    // s_hist:    16 warps × 256 bins × 1 byte = 4096 B (per-warp histograms)
    // s_tile:    192 × 32 = 6144 B (cooperative input tile)
    // s_bbox:    4 ints (min_x, min_y, max_x, max_y) for reduction
    // s_use_tile: 1 byte flag — set if bbox fits in SLOW_TILE_MAX_*
    __shared__ uint8_t  s_hist[WARPS_PER_BLOCK][256];
    __shared__ uint8_t  s_tile[SLOW_TILE_MAX_H][SLOW_TILE_MAX_W];
    __shared__ int      s_bbox[4];   // [min_gx, min_gy, max_gx, max_gy]
    __shared__ int      s_tile_origin[2];   // [origin_x, origin_y] in image coords
    __shared__ int      s_tile_w;
    __shared__ int      s_tile_h;
    __shared__ int      s_use_tile;

    const int tid = blockIdx.x * SLOW_BLOCK_SIZE + threadIdx.x;
    const int warp_id = threadIdx.x >> 5;
    uint8_t* my_hist = s_hist[warp_id];

    for (int c = 0; c < channels; c++) {
        const int n = slow_count[c];
        const bool active = (tid < n);

        // ── Read this thread's pixel coordinates (or use sentinel) ──────────
        int pixel_idx = -1, gx = -1, gy = -1;
        uint8_t Z_xy = 0;
        if (active) {
            pixel_idx = slow_list[c * area + tid];
            gx = pixel_idx % width;
            gy = pixel_idx / width;
            Z_xy = in_data[c * area + pixel_idx];
        }

        // ────────────────────────────────────────────────────────────────────
        // CRITICAL: every __syncthreads() below MUST be reached by every
        // thread in the block, regardless of `active`.  Inactive threads
        // walk through the same control flow but skip writes/reads of their
        // own pixel.  Divergent syncs cause illegal memory access on
        // GPUs without independent thread scheduling fully respecting
        // arrive-style barriers (i.e. all CUDA arch).
        // ────────────────────────────────────────────────────────────────────

        // ────────────────────────────────────────────────────────────────────
        // FIX #1 STEP 1: BLOCK-WIDE BOUNDING BOX REDUCTION
        // ────────────────────────────────────────────────────────────────────
        if (threadIdx.x == 0) {
            s_bbox[0] = INT_MAX;   // min_gx
            s_bbox[1] = INT_MAX;   // min_gy
            s_bbox[2] = -1;        // max_gx
            s_bbox[3] = -1;        // max_gy
        }
        __syncthreads();

        if (active) {
            atomicMin(&s_bbox[0], gx);
            atomicMin(&s_bbox[1], gy);
            atomicMax(&s_bbox[2], gx);
            atomicMax(&s_bbox[3], gy);
        }
        __syncthreads();

        // ────────────────────────────────────────────────────────────────────
        // FIX #1 STEP 2: DECIDE TILE / FALLBACK, COMPUTE TILE PARAMS
        // ────────────────────────────────────────────────────────────────────
        if (threadIdx.x == 0) {
            int min_gx = s_bbox[0], min_gy = s_bbox[1];
            int max_gx = s_bbox[2], max_gy = s_bbox[3];

            // SAFETY: if no thread in the block was active, max_* will be -1
            // and min_* will be INT_MAX → tile_w/tile_h would be garbage.
            // In that case mark the tile invalid (no thread will use it).
            if (max_gx < 0 || max_gy < 0) {
                s_use_tile = 0;
                s_tile_origin[0] = 0;
                s_tile_origin[1] = 0;
                s_tile_w = 0;
                s_tile_h = 0;
            } else {
                int origin_x = max(0, min_gx - R_MAX);
                int origin_y = max(0, min_gy - R_MAX);
                int far_x    = min(width  - 1, max_gx + R_MAX);
                int far_y    = min(height - 1, max_gy + R_MAX);

                int tile_w = far_x - origin_x + 1;
                int tile_h = far_y - origin_y + 1;

                s_use_tile = (tile_w > 0 && tile_h > 0 &&
                              tile_w <= SLOW_TILE_MAX_W &&
                              tile_h <= SLOW_TILE_MAX_H) ? 1 : 0;
                s_tile_origin[0] = origin_x;
                s_tile_origin[1] = origin_y;
                s_tile_w = tile_w;
                s_tile_h = tile_h;
            }
        }
        __syncthreads();

        const int use_tile = s_use_tile;
        const int origin_x = s_tile_origin[0];
        const int origin_y = s_tile_origin[1];
        const int tile_w   = s_tile_w;
        const int tile_h   = s_tile_h;

        // ────────────────────────────────────────────────────────────────────
        // FIX #1 STEP 3: COOPERATIVE TILE LOAD (all threads participate)
        // ────────────────────────────────────────────────────────────────────
        if (use_tile) {
            const int total_tile = tile_w * tile_h;
            for (int i = threadIdx.x; i < total_tile; i += SLOW_BLOCK_SIZE) {
                int tr = i / tile_w;
                int tc = i % tile_w;
                int gr = origin_y + tr;
                int gc = origin_x + tc;
                s_tile[tr][tc] = in_data[c * area + gr * width + gc];
            }
        }
        __syncthreads();   // ALL threads always reach this barrier

        // ────────────────────────────────────────────────────────────────────
        // PER-PIXEL COMPUTE (only active threads do real work)
        // Inactive threads must still reach the final __syncthreads() at the
        // bottom of this for-iteration; they simply skip the histogram and
        // output write.  No sync points exist inside this 'if (active)' block.
        // ────────────────────────────────────────────────────────────────────
        if (active) {
            // Local tile-coordinate origin for this thread's pixel
            const int local_x = gx - origin_x;
            const int local_y = gy - origin_y;
            (void)local_x; (void)local_y;   // silence unused-warning if any

            // Inline accessor — reads from shared tile if available, otherwise
            // from global with __ldg.
            #define LOAD_PIXEL(GR, GC) (use_tile \
                ? s_tile[(GR) - origin_y][(GC) - origin_x] \
                : __ldg(&in_data[c * area + (GR) * width + (GC)]))

            // ── CLEAR HISTOGRAM SLICE ───────────────────────────────────────
            #pragma unroll 16
            for (int i = 0; i < 256; i++) my_hist[i] = 0;

            // ── FIX #5: INCREMENTAL HISTOGRAM ───────────────────────────────
            int window_size = MIN_WINDOW_SIZE;
            uint8_t final_color = Z_xy;

            // Iteration 0: full 3x3
            #pragma unroll
            for (int wy = -1; wy <= 1; wy++) {
                int gr = max(0, min(gy + wy, height - 1));
                #pragma unroll
                for (int wx = -1; wx <= 1; wx++) {
                    int gc = max(0, min(gx + wx, width - 1));
                    my_hist[ LOAD_PIXEL(gr, gc) ]++;
                }
            }

            bool decided = false;

            while (!decided && window_size <= S_MAX) {

                // Add new outer ring for window_size > 3
                if (window_size > 3) {
                    const int r = window_size / 2;
                    {
                        int gr_top = max(0, min(gy - r, height - 1));
                        int gr_bot = max(0, min(gy + r, height - 1));
                        for (int wx = -r; wx <= r; wx++) {
                            int gc = max(0, min(gx + wx, width - 1));
                            my_hist[ LOAD_PIXEL(gr_top, gc) ]++;
                            my_hist[ LOAD_PIXEL(gr_bot, gc) ]++;
                        }
                    }
                    {
                        int gc_lft = max(0, min(gx - r, width - 1));
                        int gc_rgt = max(0, min(gx + r, width - 1));
                        for (int wy = -r + 1; wy <= r - 1; wy++) {
                            int gr = max(0, min(gy + wy, height - 1));
                            my_hist[ LOAD_PIXEL(gr, gc_lft) ]++;
                            my_hist[ LOAD_PIXEL(gr, gc_rgt) ]++;
                        }
                    }
                }

                // ── FIX #3: TWO-PASS EARLY-EXIT SCAN ────────────────────────
                int Z_min_h = -1, Z_med_h = -1, Z_max_h = -1;
                int cum = 0;
                const int target = ((window_size * window_size) / 2) + 1;

                #pragma unroll 8
                for (int i = 0; i < 256; i++) {
                    int count = my_hist[i];
                    if (count > 0) {
                        if (Z_min_h == -1) Z_min_h = i;
                        cum += count;
                        if (cum >= target) {
                            Z_med_h = i;
                            break;
                        }
                    }
                }

                #pragma unroll 8
                for (int i = 255; i >= 0; i--) {
                    if (my_hist[i] > 0) {
                        Z_max_h = i;
                        break;
                    }
                }

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

            #undef LOAD_PIXEL
        }   // end if (active)

        // ALL threads (active or not) reach this barrier
        __syncthreads();
    }
}

// ===========================================================================
// HOST CODE (unchanged)
// ===========================================================================
uint8_t* load_image_soa(const char *filename, int *w, int *h, int *channels) {
    FILE *f = fopen(filename, "rb");
    if (!f) return NULL;
    char magic[3];
    fscanf(f, "%2s", magic);
    if      (strcmp(magic, "P5") == 0) *channels = 1;
    else if (strcmp(magic, "P6") == 0) *channels = 3;
    else { fclose(f); return NULL; }
    int max_val;
    fscanf(f, "%d %d %d", w, h, &max_val);
    fgetc(f);
    size_t area = (size_t)(*w) * (*h);
    uint8_t *raw  = (uint8_t*)malloc(area * *channels);
    uint8_t *soa  = (uint8_t*)malloc(area * *channels);
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

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <input.ppm> [output.ppm]\n", argv[0]);
        return 1;
    }
    const char *input_file  = argv[1];
    const char *output_file = (argc > 2) ? argv[2] : "output_optimized.ppm";

    int width, height, channels;
    uint8_t *h_in = load_image_soa(input_file, &width, &height, &channels);
    if (!h_in) { fprintf(stderr, "Errore caricamento immagine!\n"); return 1; }

    const int   area    = width * height;
    const size_t img_sz = (size_t)area * channels;

    uint8_t *d_in, *d_out;
    cudaCheckError(cudaMalloc(&d_in,  img_sz));
    cudaCheckError(cudaMalloc(&d_out, img_sz));
    cudaCheckError(cudaMemcpy(d_in, h_in, img_sz, cudaMemcpyHostToDevice));

    int *d_slow_list, *d_slow_count;
    cudaCheckError(cudaMalloc(&d_slow_list,  (size_t)area * channels * sizeof(int)));
    cudaCheckError(cudaMalloc(&d_slow_count, channels * sizeof(int)));
    cudaCheckError(cudaMemset(d_slow_count, 0, channels * sizeof(int)));

    dim3 fast_block(TILE_W, TILE_H);
    dim3 fast_grid((width  + TILE_W - 1) / TILE_W,
                   (height + TILE_H - 1) / TILE_H);

    printf("=== Adaptive Median Filter — Fixes 1+3+5 ===\n");
    printf("Image : %d x %d x %d ch\n", width, height, channels);

    cudaEvent_t t0, t1, t2, t3;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventCreate(&t2); cudaEventCreate(&t3);

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
        printf("  Ch %d: %d / %d pixels need slow path (%.1f%%)\n",
               c, h_slow_count[c], area, 100.0f * h_slow_count[c] / area);

    // ── SORT slow_list per channel ──────────────────────────────────────────
    cudaEventRecord(t1);
    for (int c = 0; c < channels; c++) {
        if (h_slow_count[c] == 0) continue;
        thrust::device_ptr<int> base(d_slow_list + (size_t)c * area);
        thrust::sort(thrust::device, base, base + h_slow_count[c]);
    }
    cudaEventRecord(t2);
    cudaCheckError(cudaDeviceSynchronize());

    float ms_sort = 0;
    cudaEventElapsedTime(&ms_sort, t1, t2);
    printf("Sort          : %.3f ms\n", ms_sort);

    // ── PASS 2 ──────────────────────────────────────────────────────────────
    int max_slow = 0;
    for (int c = 0; c < channels; c++)
        if (h_slow_count[c] > max_slow) max_slow = h_slow_count[c];

    if (max_slow > 0) {
        int slow_grid = (max_slow + SLOW_BLOCK_SIZE - 1) / SLOW_BLOCK_SIZE;
        cudaEventRecord(t2);
        slow_pass_kernel<<<slow_grid, SLOW_BLOCK_SIZE>>>(
            d_in, d_out, width, height, channels,
            d_slow_list, d_slow_count, area);
        cudaCheckError(cudaPeekAtLastError());
        cudaEventRecord(t3);
        cudaCheckError(cudaDeviceSynchronize());

        float ms_slow = 0;
        cudaEventElapsedTime(&ms_slow, t2, t3);
        printf("Pass 2 (slow) : %.3f ms  (%d threads)\n", ms_slow, max_slow);
    } else {
        cudaEventRecord(t3);
    }

    float ms_total = 0;
    cudaEventElapsedTime(&ms_total, t0, t3);
    printf("Total GPU time: %.3f ms\n", ms_total);

    uint8_t *h_out = (uint8_t*)malloc(img_sz);
    cudaCheckError(cudaMemcpy(h_out, d_out, img_sz, cudaMemcpyDeviceToHost));
    save_image_soa(output_file, h_out, width, height, channels);
    printf("Salvato in %s\n", output_file);

    cudaEventDestroy(t0); cudaEventDestroy(t1);
    cudaEventDestroy(t2); cudaEventDestroy(t3);
    cudaFree(d_in); cudaFree(d_out);
    cudaFree(d_slow_list); cudaFree(d_slow_count);
    free(h_in); free(h_out);
    return 0;
}

