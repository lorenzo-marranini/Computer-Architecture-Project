#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>

// ---------------------------------------------------------------------------
// CONFIGURAZIONE
// ---------------------------------------------------------------------------
#define S_MAX           15
#define MIN_WINDOW_SIZE 3
#define R_MAX           (S_MAX / 2)

#define TILE_W          32
#define TILE_H          8
#define SHARED_PITCH    64 
#define SHARED_H        (TILE_H + 2 * R_MAX)
#define THREADS_PER_BLOCK (TILE_W * TILE_H)

#define SLOW_BLOCK_SIZE 256
#define WARPS_PER_BLOCK (SLOW_BLOCK_SIZE / 32)

#define SWAP_U8(a, b) { uint8_t tmp = min(a,b); b = max(a,b); a = tmp; }

#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// ===========================================================================
// KERNEL 1 — FAST PASS (Vectorized 16-byte Loads)
// ===========================================================================
__global__ void fast_pass_kernel(
    const uint8_t* __restrict__ in_data, uint8_t* __restrict__ out_data,
    size_t pitch, int width, int height, int channels,
    int* __restrict__ slow_list, int* __restrict__ slow_count, int area)
{
    __shared__ __align__(16) uint8_t s_in[SHARED_H][SHARED_PITCH];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x * TILE_W;
    const int by = blockIdx.y * TILE_H;
    const int gx = bx + tx;
    const int gy = by + ty;
    const int flat_tid = ty * TILE_W + tx;

    for (int c = 0; c < channels; c++) {
        const uint8_t* chan_in = in_data + c * (pitch * height);

        // Caricamento vettorializzato
        const int total_uint4 = (SHARED_H * SHARED_PITCH) / 16;
        for (int i = flat_tid; i < total_uint4; i += THREADS_PER_BLOCK) {
            int s_row = (i * 16) / SHARED_PITCH;
            int s_col = (i * 16) % SHARED_PITCH;
            int g_row = max(0, min(by + s_row - R_MAX, height - 1));
            int g_col = max(0, min(bx + s_col - R_MAX, width - 1));
            ((uint4*)s_in)[i] = *((uint4*)(chan_in + g_row * pitch + (g_col & ~0xF)));
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

            // Sorting Network 3x3
            SWAP_U8(w9[0],w9[1]); SWAP_U8(w9[3],w9[4]); SWAP_U8(w9[6],w9[7]);
            SWAP_U8(w9[1],w9[2]); SWAP_U8(w9[4],w9[5]); SWAP_U8(w9[7],w9[8]);
            SWAP_U8(w9[0],w9[1]); SWAP_U8(w9[3],w9[4]); SWAP_U8(w9[6],w9[7]);
            SWAP_U8(w9[0],w9[3]); SWAP_U8(w9[1],w9[4]); SWAP_U8(w9[2],w9[5]);
            SWAP_U8(w9[3],w9[6]); SWAP_U8(w9[4],w9[7]); SWAP_U8(w9[5],w9[8]);
            SWAP_U8(w9[0],w9[3]); SWAP_U8(w9[1],w9[4]); SWAP_U8(w9[2],w9[5]);
            SWAP_U8(w9[2],w9[6]); SWAP_U8(w9[1],w9[3]); SWAP_U8(w9[5],w9[7]);
            SWAP_U8(w9[2],w9[4]); SWAP_U8(w9[4],w9[6]); SWAP_U8(w9[3],w9[5]);
            SWAP_U8(w9[2],w9[3]); SWAP_U8(w9[4],w9[5]);

            const uint8_t z_min = w9[0], z_max = w9[8], z_med = w9[4];
            const int pixel_idx = gy * width + gx;

            if ((z_med - z_min) > 0 && (z_med - z_max) < 0) {
                out_data[c * (pitch * height) + gx + gy * pitch] = 
                    ((Z_xy - z_min) > 0 && (Z_xy - z_max) < 0) ? Z_xy : z_med;
            } else {
                int slot = atomicAdd(&slow_count[c], 1);
                slow_list[c * area + slot] = pixel_idx;
            }
        }
        __syncthreads();
    }
}

// ===========================================================================
// KERNEL 2 — SLOW PASS (Warp-Cooperative: 1 Warp per Pixel)
// ===========================================================================
__global__ void slow_pass_kernel_optimized(
    const uint8_t* __restrict__ in_data, uint8_t* __restrict__ out_data,
    size_t pitch, int width, int height, int channels,
    const int* __restrict__ slow_list, const int* __restrict__ slow_count, int area)
{
    __shared__ int s_warp_hist[WARPS_PER_BLOCK][256];

    const int lane_id = threadIdx.x % 32;
    const int warp_id = threadIdx.x / 32;
    // Ogni warp prende un pixel dalla lista
    const int pixel_idx_in_list = blockIdx.x * WARPS_PER_BLOCK + warp_id;

    for (int c = 0; c < channels; c++) {
        if (pixel_idx_in_list >= slow_count[c]) continue;

        const int pixel_idx = slow_list[c * area + pixel_idx_in_list];
        const int gx = pixel_idx % width;
        const int gy = pixel_idx / width;
        const uint8_t* chan_in = in_data + c * (pitch * height);

        int window_size = MIN_WINDOW_SIZE;
        uint8_t final_color = __ldg(&chan_in[gy * pitch + gx]);

        while (window_size <= S_MAX) {
            // Reset istogramma cooperativo
            #pragma unroll
            for(int i = 0; i < 8; i++) s_warp_hist[warp_id][lane_id + i*32] = 0;
            __syncwarp();

            const int r = window_size / 2;
            const int total_px = window_size * window_size;
            
            // Caricamento COALESCENTE della finestra
            for (int i = lane_id; i < total_px; i += 32) {
                int wy = (i / window_size) - r;
                int wx = (i % window_size) - r;
                int gr = max(0, min(gy + wy, height - 1));
                int gc = max(0, min(gx + wx, width - 1));
                atomicAdd(&s_warp_hist[warp_id][__ldg(&chan_in[gr * pitch + gc])], 1);
            }
            __syncwarp();

            int z_min = -1, z_max = -1, z_med = -1;
            if (lane_id == 0) {
                int cum = 0, target = (total_px / 2) + 1;
                for (int i = 0; i < 256; i++) {
                    int cnt = s_warp_hist[warp_id][i];
                    if (cnt > 0) {
                        if (z_min == -1) z_min = i;
                        z_max = i;
                        cum += cnt;
                        if (z_med == -1 && cum >= target) z_med = i;
                    }
                }
                if ((z_med - z_min) > 0 && (z_med - z_max) < 0) final_color = (uint8_t)z_med;
            }
            
            int done = __shfl_sync(0xFFFFFFFF, (z_med != -1 && (z_med-z_min)>0 && (z_med-z_max)<0), 0);
            if (done) break;
            window_size += 2;
        }

        if (lane_id == 0) out_data[c * (pitch * height) + gy * pitch + gx] = final_color;
    }
}

// ===========================================================================
// HELPER FUNCTIONS (SOA Load/Save)
// ===========================================================================
uint8_t* load_image_soa(const char *filename, int *w, int *h, int *channels) {
    FILE *f = fopen(filename, "rb");
    if (!f) return NULL;
    char magic[3]; fscanf(f, "%2s", magic);
    if (strcmp(magic, "P5") == 0) *channels = 1;
    else if (strcmp(magic, "P6") == 0) *channels = 3;
    else { fclose(f); return NULL; }
    int max_val; fscanf(f, "%d %d %d", w, h, &max_val); fgetc(f);
    size_t area = (size_t)(*w) * (*h);
    uint8_t *raw = (uint8_t*)malloc(area * *channels);
    uint8_t *soa = (uint8_t*)malloc(area * *channels);
    fread(raw, 1, area * *channels, f);
    for (size_t i = 0; i < area; i++)
        for (int c = 0; c < *channels; c++) soa[c * area + i] = raw[i * (*channels) + c];
    free(raw); fclose(f); return soa;
}

void save_image_soa(const char *filename, uint8_t *soa, int w, int h, int channels, size_t pitch) {
    FILE *f = fopen(filename, "wb");
    if (!f) return;
    fprintf(f, "%s\n%d %d\n255\n", (channels == 1) ? "P5" : "P6", w, h);
    size_t area = (size_t)w * h;
    uint8_t *raw = (uint8_t*)malloc(area * channels);
    for (int c = 0; c < channels; c++) {
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                raw[(y * w + x) * channels + c] = soa[c * (pitch * h) + y * pitch + x];
            }
        }
    }
    fwrite(raw, 1, area * channels, f);
    free(raw); fclose(f);
}

// ===========================================================================
// MAIN
// ===========================================================================
int main(int argc, char *argv[]) {
    if (argc < 2) { fprintf(stderr, "Usage: %s <input.ppm> [output.ppm]\n", argv[0]); return 1; }
    const char *input_file = argv[1];
    const char *output_file = (argc > 2) ? argv[2] : "output_optimized.ppm";

    int width, height, channels;
    uint8_t *h_in = load_image_soa(input_file, &width, &height, &channels);
    if (!h_in) return 1;

    const int area = width * height;
    const size_t img_sz = (size_t)area * channels;

    uint8_t *d_in, *d_out;
    size_t d_pitch;
    cudaMallocPitch(&d_in, &d_pitch, width, height * channels);
    cudaMallocPitch(&d_out, &d_pitch, width, height * channels);
    cudaMemcpy2D(d_in, d_pitch, h_in, width, width, height * channels, cudaMemcpyHostToDevice);

    int *d_slow_list, *d_slow_count;
    cudaMalloc(&d_slow_list, img_sz * sizeof(int));
    cudaMalloc(&d_slow_count, channels * sizeof(int));
    cudaMemset(d_slow_count, 0, channels * sizeof(int));

    printf("=== Optimized Warp-Cooperative Adaptive Median ===\n");
    printf("Image: %d x %d (%d ch)\n", width, height, channels);

    cudaEvent_t t0, t1, t2;
    cudaEventCreate(&t0); cudaEventCreate(&t1); cudaEventCreate(&t2);

    // --- PASS 1 ---
    cudaEventRecord(t0);
    dim3 fast_block(TILE_W, TILE_H);
    dim3 fast_grid((width + TILE_W - 1) / TILE_W, (height + TILE_H - 1) / TILE_H);
    fast_pass_kernel<<<fast_grid, fast_block>>>(d_in, d_out, d_pitch, width, height, channels, d_slow_list, d_slow_count, area);
    cudaEventRecord(t1);
    
    int h_slow_count[3] = {0,0,0};
    cudaMemcpy(h_slow_count, d_slow_count, channels * sizeof(int), cudaMemcpyDeviceToHost);

    float ms_fast; cudaEventElapsedTime(&ms_fast, t0, t1);
    printf("Pass 1 (fast) : %.3f ms\n", ms_fast);

    // --- SORTING ---
    for (int c = 0; c < channels; c++) {
        if (h_slow_count[c] > 0) {
            thrust::device_ptr<int> d_ptr(d_slow_list + c * area);
            thrust::sort(thrust::device, d_ptr, d_ptr + h_slow_count[c]);
        }
    }

    // --- PASS 2 ---
    int max_slow = 0;
    for (int c = 0; c < channels; c++) if (h_slow_count[c] > max_slow) max_slow = h_slow_count[c];

    if (max_slow > 0) {
        cudaEventRecord(t1);
        // Poiché usiamo 1 Warp per pixel, il numero di warp necessarie è max_slow
        int num_warps = max_slow;
        int threads_per_block = SLOW_BLOCK_SIZE;
        int warps_per_block = threads_per_block / 32;
        int grid_size = (num_warps + warps_per_block - 1) / warps_per_block;

        slow_pass_kernel_optimized<<<grid_size, threads_per_block>>>(d_in, d_out, d_pitch, width, height, channels, d_slow_list, d_slow_count, area);
        cudaEventRecord(t2);
        cudaDeviceSynchronize();
        float ms_slow; cudaEventElapsedTime(&ms_slow, t1, t2);
        printf("Pass 2 (slow) : %.3f ms (%d pixels processed by warps)\n", ms_slow, max_slow);
    } else {
        cudaEventRecord(t2);
    }

    float ms_total; cudaEventElapsedTime(&ms_total, t0, t2);
    printf("Total GPU time: %.3f ms\n", ms_total);

    uint8_t *h_out = (uint8_t*)malloc(d_pitch * height * channels);
    cudaMemcpy2D(h_out, d_pitch, d_out, d_pitch, width, height * channels, cudaMemcpyDeviceToHost);
    save_image_soa(output_file, h_out, width, height, channels, d_pitch);

    cudaFree(d_in); cudaFree(d_out); cudaFree(d_slow_list); cudaFree(d_slow_count);
    free(h_in); free(h_out);
    return 0;
}