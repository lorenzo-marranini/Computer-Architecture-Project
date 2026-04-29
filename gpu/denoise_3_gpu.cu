#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <cuda_runtime.h>

// --- COSTANTI DI CONFIGURAZIONE ---
#define S_MAX 15
#define MIN_WINDOW_SIZE 3
#define R_MAX (S_MAX / 2)       // 7
#define TILE_SIZE 8             // Blocco 8x8 = 64 thread (Ottimale per la Shared Memory)
#define SHARED_SIZE (TILE_SIZE + 2 * R_MAX) // 8 + 14 = 22

// --- GESTIONE ERRORI CUDA ---
#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// ---------------------------------------------------------
// KERNEL GPU ULTRA-OTTIMIZZATO
// ---------------------------------------------------------
__global__ void adaptive_median_ultra_optimized_kernel(
    const uint8_t *__restrict__ in_data,
    uint8_t *__restrict__ out_data,
    int width, 
    int height, 
    int channels) 
{
    // Tavolo da lavoro per i pixel (22x22)
    __shared__ uint8_t s_in[SHARED_SIZE][SHARED_SIZE];
    
    // Matrice degli Istogrammi: [256 colori][64 thread]. 
    // Nessun bank conflict, tutto in Cache L1!
    __shared__ uint8_t s_buckets[256][TILE_SIZE * TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x * TILE_SIZE;
    int by = blockIdx.y * TILE_SIZE;
    int gx = bx + tx;
    int gy = by + ty;
    int tid = ty * TILE_SIZE + tx; // ID da 0 a 63

    for (int c = 0; c < channels; c++) {
        
        // ========================================================
        // FASE 1: CARICAMENTO COOPERATIVO (CON SoA MEMORY COALESCING)
        // ========================================================
        for (int i = tid; i < SHARED_SIZE * SHARED_SIZE; i += TILE_SIZE * TILE_SIZE) {
            int s_row = i / SHARED_SIZE;
            int s_col = i % SHARED_SIZE;
            int g_row = max(0, min(by + s_row - R_MAX, height - 1));
            int g_col = max(0, min(bx + s_col - R_MAX, width - 1));
            
            // Lettura Lineare SoA (Planare) - Perfetta per la VRAM
            s_in[s_row][s_col] = in_data[c * (width * height) + (g_row * width + g_col)];
        }
        __syncthreads();

        // ========================================================
        // FASE 2: ELABORAZIONE DEL PIXEL
        // ========================================================
        if (gx < width && gy < height) {
            int s_y = ty + R_MAX;
            int s_x = tx + R_MAX;
            uint8_t Z_xy = s_in[s_y][s_x];
            
            // --- OPTIMIZATION 1: IL FAST-PATH 3x3 NEI REGISTRI ---
            // Dato che il 90% dei pixel si risolve a 3x3, lo facciamo
            // senza nemmeno toccare l'istogramma, usando solo registri velocissimi.
            uint8_t local_9[9];
            int idx = 0;
            #pragma unroll
            for(int wy = -1; wy <= 1; wy++) {
                #pragma unroll
                for(int wx = -1; wx <= 1; wx++) {
                    local_9[idx++] = s_in[s_y + wy][s_x + wx];
                }
            }

            // Bubble Sort super-srotolato nei registri
            #pragma unroll
            for (int i = 0; i < 9; i++) {
                #pragma unroll
                for (int j = i + 1; j < 9; j++) {
                    if (local_9[i] > local_9[j]) {
                        uint8_t tmp = local_9[i];
                        local_9[i] = local_9[j];
                        local_9[j] = tmp;
                    }
                }
            }

            uint8_t z_min = local_9[0];
            uint8_t z_max = local_9[8];
            uint8_t z_med = local_9[4];

            // Controllo Adattivo per il 3x3
            if ((z_med - z_min) > 0 && (z_med - z_max) < 0) {
                if ((Z_xy - z_min) > 0 && (Z_xy - z_max) < 0) {
                    out_data[c * (width * height) + (gy * width + gx)] = Z_xy;
                } else {
                    out_data[c * (width * height) + (gy * width + gx)] = z_med;
                }
                // Il pixel è pulito! Usciamo subito, niente Warp Divergence inutile!
                goto FINE_PIXEL; 
            }

            // --- OPTIMIZATION 2: IL SLOW-PATH CON ISTOGRAMMA IN SHARED MEMORY ---
            // Se arriviamo qui, il pixel era rumoroso anche a 3x3. Avviamo la logica pesante.
            
            #pragma unroll 16
            for (int i = 0; i < 256; i++) {
                s_buckets[i][tid] = 0; 
            }

            int window_size = MIN_WINDOW_SIZE;
            uint8_t result_color = Z_xy;

            while (window_size <= S_MAX) {
                int r = window_size / 2;
                
                if (window_size == 3) {
                    for (int wy = -1; wy <= 1; wy++) {
                        for (int wx = -1; wx <= 1; wx++) {
                            s_buckets[ s_in[s_y + wy][s_x + wx] ][tid]++;
                        }
                    }
                } else {
                    for (int wx = -r; wx <= r; wx++) {
                        s_buckets[ s_in[s_y - r][s_x + wx] ][tid]++; 
                        s_buckets[ s_in[s_y + r][s_x + wx] ][tid]++; 
                    }
                    for (int wy = -r + 1; wy <= r - 1; wy++) {
                        s_buckets[ s_in[s_y + wy][s_x - r] ][tid]++; 
                        s_buckets[ s_in[s_y + wy][s_x + r] ][tid]++; 
                    }
                }

                int Z_min_hist = -1, Z_max_hist = -1, Z_med_hist = -1;
                int count_cumulativo = 0;
                int target_pos = ((window_size * window_size) / 2) + 1; 
                
                #pragma unroll 16
                for (int i = 0; i < 256; i++) {
                    if (s_buckets[i][tid] > 0) {
                        if (Z_min_hist == -1) Z_min_hist = i;
                        Z_max_hist = i; 
                        count_cumulativo += s_buckets[i][tid];
                        if (Z_med_hist == -1 && count_cumulativo >= target_pos) Z_med_hist = i;
                    }
                }
                
                if ((Z_med_hist - Z_min_hist) > 0 && (Z_med_hist - Z_max_hist) < 0) {
                    result_color = ((Z_xy - Z_min_hist) > 0 && (Z_xy - Z_max_hist) < 0) ? Z_xy : Z_med_hist; 
                    break; 
                } else {
                    window_size += 2;
                    if (window_size > S_MAX) { result_color = Z_med_hist; break; }
                }
            }
            out_data[c * (width * height) + (gy * width + gx)] = result_color;
            
            FINE_PIXEL:; // Etichetta per il goto del fast-path
        }
        // Sincronizzazione obbligatoria prima di cambiare canale colore
        __syncthreads();
    }
}

// ---------------------------------------------------------
// FUNZIONI HOST (CPU)
// ---------------------------------------------------------

// Modificata per convertire l'immagine da AoS (RGBRGB...) a SoA (RRR...GGG...BBB...)
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
    
    // Converti in Planare (SoA)
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

// Modificata per riconvertire l'immagine da SoA ad AoS per il salvataggio
void save_image_soa(const char *filename, uint8_t *soa_data, int w, int h, int channels) {
    FILE *f = fopen(filename, "wb");
    if (!f) return;
    const char *magic = (channels == 1) ? "P5" : "P6";
    fprintf(f, "%s\n%d %d\n255\n", magic, w, h);
    
    size_t area = (size_t)w * h;
    uint8_t *raw_data = (uint8_t*)malloc(area * channels);
    
    // Riconverti in Interleaved (AoS)
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
    
    // Griglia aggiornata per blocchi 8x8
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 numBlocks((width + TILE_SIZE - 1) / TILE_SIZE,
                   (height + TILE_SIZE - 1) / TILE_SIZE);

    printf("Avvio Filtro GPU Definitivo (SoA + Shared Matrix + Fast-Path)...\n");

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