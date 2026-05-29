// Versione Ottimizzata con Global Memory RESTRICT


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <stdint.h>
const int S_MAX = 15;
const int MIN_WINDOW_SIZE = 3;


// Macro per il controllo degli errori CUDA
#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// ---------------------------------------------------------
// MACRO CUDA: Simili alle tue, ma adattate per il device
// ---------------------------------------------------------
#define ADD_PIXEL_SAFE_CUDA(wy, wx) \
    buckets[in_data[( (y + (wy)) * width + (x + (wx)) ) * channels + c]]++


#define ADD_PIXEL_CLAMPED_CUDA(wy, wx) \
    do { \
        /* Uso delle funzioni matematiche native max/min di CUDA per evitare branch (if) */ \
        int ny = max(0, min(y + (wy), height - 1)); \
        int nx = max(0, min(x + (wx), width - 1)); \
        buckets[in_data[(ny * width + nx) * channels + c]]++; \
    } while(0)


// ---------------------------------------------------------
// KERNEL CUDA (Sostituisce il tuo worker pthread)
// ---------------------------------------------------------
__global__ void adaptive_median_optimized_kernel(
    const unsigned char *__restrict__ in_data,
    unsigned char *__restrict__ out_data,
    int width, 
    int height, 
    int channels) 
{
    // Il Dynamic Load Balancer (chunk) non serve più. 
    // La GPU assegna automaticamente a questo thread un singolo pixel (x, y).
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Controllo strettamente necessario per i thread che sbordano dall'immagine
    if (x >= width || y >= height) return;

    uint8_t buckets[256];
    int max_radius = S_MAX / 2; 

    // Questa variabile booleana determina se il thread corrente può evitare i controlli.
    // Vantaggio GPU: Se tutti i 32 thread di un Warp sono "safe", eseguiranno il codice
    // senza alcuna divergenza o rallentamento.
    bool is_safe = (y >= max_radius && y < height - max_radius && 
                    x >= max_radius && x < width - max_radius);

    for (int c = 0; c < channels; c++) {
        
        int out_idx = (y * width + x) * channels + c;
        unsigned char Z_xy = in_data[out_idx];
        
        // Inizializzazione pulita dell'istogramma (sostituisce memset)
        for (int i = 0; i < 256; i++) buckets[i] = 0;
        
        int window_size = MIN_WINDOW_SIZE;
        unsigned char result_color = Z_xy;

        while (window_size <= S_MAX) {
            
            int r = window_size / 2;
            
            // Logica esatta della tua versione CPU riportata su GPU
            if (is_safe) {
                if (window_size == 3) {
                    // Finestra iniziale 3x3
                    for (int wy = -1; wy <= 1; wy++) {
                        for (int wx = -1; wx <= 1; wx++) ADD_PIXEL_SAFE_CUDA(wy, wx);
                    }
                } else {
                    // Aggiornamento Incrementale: aggiungo solo le cornici
                    // Righe (Sopra e Sotto)
                    for (int wx = -r; wx <= r; wx++) {
                        ADD_PIXEL_SAFE_CUDA(-r, wx); 
                        ADD_PIXEL_SAFE_CUDA(r, wx);  
                    }
                    // Colonne (Sinistra e Destra, escludendo gli angoli già letti)
                    for (int wy = -r + 1; wy <= r - 1; wy++) {
                        ADD_PIXEL_SAFE_CUDA(wy, -r);
                        ADD_PIXEL_SAFE_CUDA(wy, r);                                
                    }
                }
            } else {
                // Versione per i bordi: usa max/min invece di if espliciti
                if (window_size == 3) {
                    for (int wy = -1; wy <= 1; wy++) {
                        for (int wx = -1; wx <= 1; wx++) ADD_PIXEL_CLAMPED_CUDA(wy, wx);
                    }
                } else {
                    for (int wx = -r; wx <= r; wx++) {
                        ADD_PIXEL_CLAMPED_CUDA(-r, wx); 
                        ADD_PIXEL_CLAMPED_CUDA(r, wx);  
                    }
                    for (int wy = -r + 1; wy <= r - 1; wy++) {
                        ADD_PIXEL_CLAMPED_CUDA(wy, -r); 
                        ADD_PIXEL_CLAMPED_CUDA(wy, r);  
                    }
                }
            }

            // Calcolo del mediano dai buckets (identico alla CPU)
            int Z_min = -1, Z_max = -1, Z_med = -1;
            int count_cumulativo = 0;
            int target_pos = ((window_size * window_size) / 2) + 1; 
            
            for (int i = 0; i < 256; i++) {
                if (buckets[i] > 0) {
                    if (Z_min == -1) Z_min = i;
                    Z_max = i; 
                    count_cumulativo += buckets[i];
                    if (Z_med == -1 && count_cumulativo >= target_pos) {
                        Z_med = i;
                    }
                }
            }
            
            // Logica Adattiva
            if ((Z_med - Z_min) > 0 && (Z_med - Z_max) < 0) {
                if ((Z_xy - Z_min) > 0 && (Z_xy - Z_max) < 0) {
                    result_color = Z_xy; 
                } else {
                    result_color = Z_med; 
                }
                break; 
            } else {
                window_size += 2;
                if (window_size > S_MAX) {
                    result_color = Z_med; 
                    break;
                }
            }
        }
        out_data[out_idx] = result_color;
    }
}


unsigned char* load_image(const char *filename, int *w, int *h, int *channels) {
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
    unsigned char *data = (unsigned char*)malloc(*w * *h * *channels);
    fread(data, 1, *w * *h * *channels, f);
    fclose(f);
    return data;
}

void save_image(const char *filename, unsigned char *data, int w, int h, int channels) {
    FILE *f = fopen(filename, "wb");
    if (!f) return;
    const char *magic = (channels == 1) ? "P5" : "P6";
    fprintf(f, "%s\n%d %d\n255\n", magic, w, h);
    fwrite(data, 1, w * h * channels, f);
    fclose(f);
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <input.ppm> [output.ppm]\n", argv[0]);
        return 1;
    }
    
    const char *input_file = argv[1];
    const char *output_file = (argc > 2) ? argv[2] : "output.ppm";
    
    int width, height, channels;
    unsigned char *h_img_in = load_image(input_file, &width, &height, &channels);
    if (!h_img_in) return 1;
    
    size_t img_size = width * height * channels * sizeof(unsigned char);
    unsigned char *h_img_out = (unsigned char*)malloc(img_size);
    
    // Allocazione VRAM
    unsigned char *d_img_in, *d_img_out;
    cudaCheckError(cudaMalloc((void**)&d_img_in, img_size));
    cudaCheckError(cudaMalloc((void**)&d_img_out, img_size));
    
    // Trasferimento Host -> Device
    cudaCheckError(cudaMemcpy(d_img_in, h_img_in, img_size, cudaMemcpyHostToDevice));
    
    // Configurazione Griglia.
    // BLOCK_X / BLOCK_Y sono overridabili a compile-time con -D per sweep.
    // Default: 32×8 = 256 thread/block (8 warp).
    #ifndef BLOCK_X
    #define BLOCK_X 32
    #endif
    #ifndef BLOCK_Y
    #define BLOCK_Y 4
    #endif
    dim3 threadsPerBlock(BLOCK_X, BLOCK_Y);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    printf("Filtro Mediano Adattivo GPU Ottimizzato avviato...\n");
    cudaEventRecord(start); // Fai partire il timer

    // Lancio Kernel
    adaptive_median_optimized_kernel<<<numBlocks, threadsPerBlock>>>(d_img_in, d_img_out, width, height, channels);
    
    
    cudaEventRecord(stop); // Ferma il timer

    cudaCheckError(cudaPeekAtLastError());
    cudaCheckError(cudaDeviceSynchronize());
    
    // Calcola e stampa tempo
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Total GPU time %.3f ms\n", milliseconds);
    

    // Trasferimento Device -> Host
    cudaCheckError(cudaMemcpy(h_img_out, d_img_out, img_size, cudaMemcpyDeviceToHost));
    
    // Salvataggio (scommenta nel tuo codice)
    save_image(output_file, h_img_out, width, height, channels);
    printf("Completato. Salvato in %s\n", output_file);
    
    cudaFree(d_img_in);
    cudaFree(d_img_out);
    free(h_img_in);
    free(h_img_out);
    
    return 0;
}
