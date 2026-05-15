#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <cuda_runtime.h>

// --- COSTANTI DI CONFIGURAZIONE ---
#define S_MAX 15
#define MIN_WINDOW_SIZE 3
#define R_MAX (S_MAX / 2)       // 7

// --- OTTIMIZZAZIONI GRIGLIA ---
#define TILE_W 32               // Warp perfetto (32 thread)
#define TILE_H 8

// --- MACRO HARDWARE ---
// Sorting Network Branchless: scambia due valori sfruttando le istruzioni hardware min/max
#define SWAP_U8(a, b) { uint8_t tmp = min(a, b); b = max(a, b); a = tmp; }

// --- ACCESSO DIRETTO ALLA GLOBAL MEMORY (sostituisce s_in[][]) ---
// Stessa logica branchless di clamping già usata nel caricamento cooperativo
// della versione con shared memory: max/min invece di if espliciti.
// channel_base = c * (width * height) → offset del piano SoA del canale c.
#define READ_GLOBAL(wy, wx) \
    in_data[channel_base + max(0, min(gy + (wy), height - 1)) * width \
                         + max(0, min(gx + (wx), width  - 1))]

// --- GESTIONE ERRORI CUDA ---
#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// ---------------------------------------------------------
// KERNEL GPU — NO SHARED MEMORY
// ---------------------------------------------------------
// Identico alla versione "Occupancy 100% + Coalescing + Sorting Network"
// salvo la rimozione della Fase 1 (caricamento cooperativo in s_in[][]).
// Ogni accesso a s_in[s_y + dy][s_x + dx] è stato sostituito con
// READ_GLOBAL(dy, dx), che legge direttamente da in_data passando per L1.
// ---------------------------------------------------------
__global__ void adaptive_median_ultra_optimized_kernel(
    const uint8_t *__restrict__ in_data,
    uint8_t *__restrict__ out_data,
    int width,
    int height,
    int channels)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x * blockDim.x;
    int by = blockIdx.y * blockDim.y;
    int gx = bx + tx;
    int gy = by + ty;

    for (int c = 0; c < channels; c++) {

        const int channel_base = c * (width * height);

        // (Fase 1 rimossa: niente caricamento cooperativo, niente __syncthreads)

        // ========================================================
        // FASE 2: ELABORAZIONE DEL PIXEL
        // ========================================================
        if (gx < width && gy < height) {
            uint8_t Z_xy = READ_GLOBAL(0, 0);

            uint8_t final_color = Z_xy;

            // --- FAST-PATH 3x3 NEI REGISTRI ---
            uint8_t local_9[9];
            int idx = 0;
            #pragma unroll
            for(int wy = -1; wy <= 1; wy++) {
                #pragma unroll
                for(int wx = -1; wx <= 1; wx++) {
                    local_9[idx++] = READ_GLOBAL(wy, wx);
                }
            }

            // ALU FIX: Sorting Network a 9 elementi al posto del Bubble Sort.
            // Nessun 'if', nessuna divergenza dei warp, massima velocità hardware.
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
                if ((Z_xy - z_min) > 0 && (Z_xy - z_max) < 0) {
                    final_color = Z_xy;
                } else {
                    final_color = z_med;
                }
            }
            else
            {
                // --- SLOW-PATH CON ISTOGRAMMA IN LOCAL MEMORY ---
                // L1 CACHE FIX: uint8_t salva il 75% della memoria ed evita lo spilling sulla DRAM lenta!
                uint8_t local_hist[256] = {0};

                int window_size = MIN_WINDOW_SIZE;

                while (window_size <= S_MAX) {
                    int r = window_size / 2;

                    if (window_size == 3) {
                        for (int wy = -1; wy <= 1; wy++) {
                            for (int wx = -1; wx <= 1; wx++) {
                                local_hist[ READ_GLOBAL(wy, wx) ]++;
                            }
                        }
                    } else {
                        for (int wx = -r; wx <= r; wx++) {
                            local_hist[ READ_GLOBAL(-r, wx) ]++;
                            local_hist[ READ_GLOBAL( r, wx) ]++;
                        }
                        for (int wy = -r + 1; wy <= r - 1; wy++) {
                            local_hist[ READ_GLOBAL(wy, -r) ]++;
                            local_hist[ READ_GLOBAL(wy,  r) ]++;
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

            // Scrittura coalescente garantita dal calcolo unificato e dal warp 32x8
            out_data[channel_base + gy * width + gx] = final_color;
        }
    }
}

// ---------------------------------------------------------
// FUNZIONI HOST E MAIN (Convertitore SoA)
// ---------------------------------------------------------
uint8_t* load_image_soa(const char *filename, int *w, int *h, int *channels) {
    FILE *f = fopen(filename, "rb");
    if (!f) return NULL;
    char magic[3];
    fscanf(f, "%2s", magic);
    if (strcmp(magic, "P5") == 0) *channels = 1;
    else if (strcmp(magic, "P6") == 0) *channels = 3;
    else { fclose(f); return NULL; }

    int max_val;
    fscanf(f, "%d %d %d", w, h, &max_val);
    fgetc(f);

    size_t area = (size_t)(*w) * (*h);
    uint8_t *raw_data = (uint8_t*)malloc(area * *channels);
    fread(raw_data, 1, area * *channels, f);

    uint8_t *soa_data = (uint8_t*)malloc(area * *channels);
    for (size_t i = 0; i < area; i++) {
        for (int c = 0; c < *channels; c++) {
            soa_data[c * area + i] = raw_data[i * (*channels) + c];
        }
    }

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

    for (size_t i = 0; i < area; i++) {
        for (int c = 0; c < channels; c++) {
            raw_data[i * channels + c] = soa_data[c * area + i];
        }
    }

    fwrite(raw_data, 1, area * channels, f);
    free(raw_data);
    fclose(f);
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <input.ppm> [output.ppm]\n", argv[0]);
        return 1;
    }
    const char *input_file = argv[1];
    const char *output_file = (argc > 2) ? argv[2] : "output_optimized.ppm";

    int width, height, channels;
    uint8_t *h_img_in = load_image_soa(input_file, &width, &height, &channels);

    if (!h_img_in) {
        fprintf(stderr, "Errore caricamento immagine!\n");
        return 1;
    }

    size_t img_size = width * height * channels * sizeof(uint8_t);
    uint8_t *h_img_out = (uint8_t*)malloc(img_size);

    uint8_t *d_img_in, *d_img_out;
    cudaCheckError(cudaMalloc((void**)&d_img_in, img_size));
    cudaCheckError(cudaMalloc((void**)&d_img_out, img_size));

    cudaCheckError(cudaMemcpy(d_img_in, h_img_in, img_size, cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(TILE_W, TILE_H);
    dim3 numBlocks((width + TILE_W - 1) / TILE_W,
                   (height + TILE_H - 1) / TILE_H);

    printf("Avvio Filtro GPU (No Shared + Coalescing + Sorting Network)...\n");

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    adaptive_median_ultra_optimized_kernel<<<numBlocks, threadsPerBlock>>>(d_img_in, d_img_out, width, height, channels);

    cudaEventRecord(stop);

    cudaCheckError(cudaPeekAtLastError());
    cudaCheckError(cudaDeviceSynchronize());

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Tempo di esecuzione GPU: %.3f ms\n", milliseconds);

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
