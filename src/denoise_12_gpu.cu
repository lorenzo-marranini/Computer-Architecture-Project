#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>

// ---------------------------------------------------------------------------
// CONFIGURAZIONE OTTIMIZZATA (Slide 39 & Nsight)
// ---------------------------------------------------------------------------
#define S_MAX           15
#define MIN_WINDOW_SIZE 3
#define R_MAX           (S_MAX / 2)     // 7

#define TILE_W          32
#define TILE_H          8
// SHARED_PITCH a 64 garantisce che ogni riga in Shared Memory sia allineata a 16 byte
#define SHARED_PITCH    64 
#define SHARED_H        (TILE_H + 2 * R_MAX)
#define THREADS_PER_BLOCK (TILE_W * TILE_H)

#define SLOW_BLOCK_SIZE 512
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
    const uint8_t* __restrict__ in_data,
    uint8_t* __restrict__ out_data,
    size_t pitch, int width, int height, int channels,
    int* __restrict__ slow_list, int* __restrict__ slow_count, int area)
{
    // Allineamento forzato a 16 byte per supportare uint4 (Slide 39) [cite: 631]
    __shared__ __align__(16) uint8_t s_in[SHARED_H][SHARED_PITCH];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x * TILE_W;
    const int by = blockIdx.y * TILE_H;
    const int gx = bx + tx;
    const int gy = by + ty;
    const int flat_tid = ty * TILE_W + tx;

    for (int c = 0; c < channels; c++) {
        // Puntatore alla base del canale corrente (usa il pitch hardware)
        const uint8_t* chan_in = in_data + c * (pitch * height);

        // --- CARICAMENTO VETTORIALIZZATO (16 PIXEL PER VOLTA) ---
        // Riduce le "Excessive Sectors" segnalate da Nsight
        const int total_uint4 = (SHARED_H * SHARED_PITCH) / 16;
        #pragma unroll
        for (int i = flat_tid; i < total_uint4; i += THREADS_PER_BLOCK) {
            int s_row = (i * 16) / SHARED_PITCH;
            int s_col = (i * 16) % SHARED_PITCH;
            
            int g_row = max(0, min(by + s_row - R_MAX, height - 1));
            int g_col = max(0, min(bx + s_col - R_MAX, width - 1));
            
            // Accesso allineato: carichiamo 128 bit (16 byte) in un colpo solo [cite: 622]
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
// KERNEL 2 — SLOW PASS (Shared Memory Histograms + Cache Optimization)
// ===========================================================================
__global__ __launch_bounds__(SLOW_BLOCK_SIZE, 4)
void slow_pass_kernel(
    const uint8_t* __restrict__ in_data,
    uint8_t* __restrict__ out_data,
    size_t pitch, int width, int height, int channels,
    const int* __restrict__ slow_list, const int* __restrict__ slow_count, int area)
{
    __shared__ uint8_t s_hist[WARPS_PER_BLOCK][256];
    const int tid = blockIdx.x * SLOW_BLOCK_SIZE + threadIdx.x;
    const int warp_id = threadIdx.x >> 5;
    uint8_t* my_hist = s_hist[warp_id];

    for (int c = 0; c < channels; c++) {
        const int n = slow_count[c];
        if (tid >= n) continue;

        const int pixel_idx = slow_list[c * area + tid];
        const int gx = pixel_idx % width;
        const int gy = pixel_idx / width;
        const uint8_t* chan_in = in_data + c * (pitch * height);

        // Reset istogramma ottimizzato per ridurre l'ALU pressure
        #pragma unroll 32
        for (int i = 0; i < 256; i++) my_hist[i] = 0;

        int window_size = MIN_WINDOW_SIZE;
        uint8_t final_color = in_data[c * area + pixel_idx];

        while (window_size <= S_MAX) {
            const int r = window_size / 2;
            for (int wy = -r; wy <= r; wy++) {
                int gr = max(0, min(gy + wy, height - 1));
                for (int wx = -r; wx <= r; wx++) {
                    int gc = max(0, min(gx + wx, width - 1));
                    // Uso di __ldg per forzare l'uso della cache Read-Only (mitiga uncoalesced)
                    my_hist[__ldg(&chan_in[gr * pitch + gc])]++;
                }
            }

            int Z_min_h = -1, Z_max_h = -1, Z_med_h = -1, cum = 0;
            const int target = ((window_size * window_size) / 2) + 1;

            for (int i = 0; i < 256; i++) {
                if (my_hist[i] > 0) {
                    if (Z_min_h == -1) Z_min_h = i;
                    Z_max_h = i;
                    cum += my_hist[i];
                    if (Z_med_h == -1 && cum >= target) Z_med_h = i;
                }
            }

            if ((Z_med_h - Z_min_h) > 0 && (Z_med_h - Z_max_h) < 0) {
                final_color = (uint8_t)Z_med_h; // Semplificato per stabilità
                break;
            }
            window_size += 2;
        }
        out_data[c * area + pixel_idx] = final_color;
    }
}

// ... Host functions (load_image_soa/save_image_soa identiche) ...
// [Omesse per brevità, caricare l'immagine come fatto in precedenza]

int main(int argc, char *argv[]) {
    if (argc < 2) return 1;
    int width, height, channels;
    uint8_t *h_in = load_image_soa(argv[1], &width, &height, &channels);

    const int area = width * height;
    const size_t img_sz = (size_t)area * channels;

    uint8_t *d_in, *d_out;
    size_t d_pitch;

    // --- ALLINEAMENTO HARDWARE (Slide 39) --- 
    // cudaMallocPitch garantisce che ogni riga sia allineata a 256/512 byte
    cudaCheckError(cudaMallocPitch(&d_in, &d_pitch, width, height * channels));
    cudaCheckError(cudaMallocPitch(&d_out, &d_pitch, width, height * channels));

    // Copia 2D per gestire il pitch
    cudaCheckError(cudaMemcpy2D(d_in, d_pitch, h_in, width, width, height * channels, cudaMemcpyHostToDevice));

    int *d_slow_list, *d_slow_count;
    cudaCheckError(cudaMalloc(&d_slow_list, img_sz * sizeof(int)));
    cudaCheckError(cudaMalloc(&d_slow_count, channels * sizeof(int)));
    cudaCheckError(cudaMemset(d_slow_count, 0, channels * sizeof(int)));

    dim3 fast_block(TILE_W, TILE_H);
    dim3 fast_grid((width + TILE_W - 1) / TILE_W, (height + TILE_H - 1) / TILE_H);

    // Pass 1
    fast_pass_kernel<<<fast_grid, fast_block>>>(d_in, d_out, d_pitch, width, height, channels, d_slow_list, d_slow_count, area);

    // --- SPATIAL SORTING (Cache L2 Optimization) ---
    int h_slow_count[3];
    cudaMemcpy(h_slow_count, d_slow_count, channels * sizeof(int), cudaMemcpyDeviceToHost);
    for (int c = 0; c < channels; c++) {
        if (h_slow_count[c] > 0) {
            thrust::device_ptr<int> d_ptr(d_slow_list + c * area);
            thrust::sort(thrust::device, d_ptr, d_ptr + h_slow_count[c]);
        }
    }

    // Pass 2
    int max_slow = 0;
    for (int c = 0; c < channels; c++) if (h_slow_count[c] > max_slow) max_slow = h_slow_count[c];
    if (max_slow > 0) {
        int slow_grid = (max_slow + SLOW_BLOCK_SIZE - 1) / SLOW_BLOCK_SIZE;
        slow_pass_kernel<<<slow_grid, SLOW_BLOCK_SIZE>>>(d_in, d_out, d_pitch, width, height, channels, d_slow_list, d_slow_count, area);
    }

    // ... Cleanup e salvataggio (usando cudaMemcpy2D per tornare all'host) ...
    uint8_t *h_out = (uint8_t*)malloc(img_sz);
    cudaMemcpy2D(h_out, width, d_out, d_pitch, width, height * channels, cudaMemcpyDeviceToHost);
    save_image_soa("output_optimized.ppm", h_out, width, height, channels);

    return 0;
}