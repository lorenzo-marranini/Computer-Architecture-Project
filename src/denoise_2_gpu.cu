// Versione Con Shared Memory con 30x30 pixel per blocco


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

// Definizione fissa a compile-time (necessaria per dimensionare la Shared Memory)
#define S_MAX 15
#define MIN_WINDOW_SIZE 3
#define R_MAX (S_MAX / 2)       // 7
#define TILE_SIZE 16            // Dimensione del blocco (16x16)
#define SHARED_SIZE (TILE_SIZE + 2 * R_MAX) // 16 + 14 = 30

// Macro errori CUDA
#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// ---------------------------------------------------------
// KERNEL CON MEMORIA CONDIVISA
// ---------------------------------------------------------
__global__ void adaptive_median_shared_kernel(
    const unsigned char *__restrict__ in_data,
    unsigned char *__restrict__ out_data,
    int width, 
    int height, 
    int channels) 
{
    // Allocazione del "tavolo da lavoro" per questo blocco.
    // Matrice 30x30 = 900 byte. Siccome ogni blocco elabora un canale alla volta,
    // usiamo la stessa matrice per R, poi G, poi B.
    __shared__ unsigned char s_in[SHARED_SIZE][SHARED_SIZE];

    // Coordinate locali del thread all'interno del blocco (0-15)
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Coordinata globale in alto a sinistra del blocco intero
    int bx = blockIdx.x * TILE_SIZE;
    int by = blockIdx.y * TILE_SIZE;

    // Coordinata globale del singolo pixel
    int gx = bx + tx;
    int gy = by + ty;
    
    // ID lineare del thread nel blocco (0-255)
    int tid = ty * TILE_SIZE + tx;

    // Elaboriamo un canale colore alla volta
    for (int c = 0; c < channels; c++) {
        
        // ========================================================
        // FASE 1: CARICAMENTO COOPERATIVO CON CLAMPING
        // ========================================================
        // Il blocco ha 256 thread, ma dobbiamo caricare 900 pixel.
        // I thread faranno "più viaggi" (ciclando) per riempire tutto l'alone.
        for (int i = tid; i < SHARED_SIZE * SHARED_SIZE; i += TILE_SIZE * TILE_SIZE) {
            int s_row = i / SHARED_SIZE; // Riga nella memoria condivisa
            int s_col = i % SHARED_SIZE; // Colonna nella memoria condivisa

            // Calcolo coordinata globale corrispondente
            int g_row = by + s_row - R_MAX;
            int g_col = bx + s_col - R_MAX;

            // Clamping Anticipato! Lo facciamo una volta sola in fase di caricamento
            g_row = max(0, min(g_row, height - 1));
            g_col = max(0, min(g_col, width - 1));

            // Scrittura sul tavolo da lavoro
            s_in[s_row][s_col] = in_data[(g_row * width + g_col) * channels + c];
        }

        // Aspettiamo che tutti i 256 thread abbiano finito di apparecchiare
        __syncthreads();

        // ========================================================
        // FASE 2: CALCOLO DEL MEDIANO (Nessun Branch!)
        // ========================================================
        // Assicuriamoci che il thread stia dentro l'immagine
        if (gx < width && gy < height) {
            
            // Il centro di questo thread nella Shared Memory è sfalsato del Raggio
            int s_y = ty + R_MAX;
            int s_x = tx + R_MAX;

            int buckets[256];
            for (int i = 0; i < 256; i++) buckets[i] = 0;

            int window_size = MIN_WINDOW_SIZE;
            unsigned char Z_xy = s_in[s_y][s_x]; // Leggiamo il pixel centrale dalla Shared!
            unsigned char result_color = Z_xy;

            while (window_size <= S_MAX) {
                int r = window_size / 2;
                
                // NOTA LA MAGIA: Niente più 'is_safe'. Nessun if o max/min.
                // Tutto l'alone è già stato preparato e "messo in sicurezza" (clamped)
                // nella Fase 1. Leggiamo tutto ciecamente dalla Shared Memory.
                if (window_size == 3) {
                    for (int wy = -1; wy <= 1; wy++) {
                        for (int wx = -1; wx <= 1; wx++) buckets[s_in[s_y + wy][s_x + wx]]++;
                    }
                } else {
                    // Cornici incrementali dalla Shared Memory
                    for (int wx = -r; wx <= r; wx++) {
                        buckets[s_in[s_y - r][s_x + wx]]++; 
                        buckets[s_in[s_y + r][s_x + wx]]++; 
                    }
                    for (int wy = -r + 1; wy <= r - 1; wy++) {
                        buckets[s_in[s_y + wy][s_x - r]]++; 
                        buckets[s_in[s_y + wy][s_x + r]]++; 
                    }
                }

                // Calcolo Mediano (Identico a prima)
                int Z_min = -1, Z_max = -1, Z_med = -1;
                int count_cumulativo = 0;
                int target_pos = ((window_size * window_size) / 2) + 1; 
                
                for (int i = 0; i < 256; i++) {
                    if (buckets[i] > 0) {
                        if (Z_min == -1) Z_min = i;
                        Z_max = i; 
                        count_cumulativo += buckets[i];
                        if (Z_med == -1 && count_cumulativo >= target_pos) Z_med = i;
                    }
                }
                
                if ((Z_med - Z_min) > 0 && (Z_med - Z_max) < 0) {
                    if ((Z_xy - Z_min) > 0 && (Z_xy - Z_max) < 0) result_color = Z_xy; 
                    else result_color = Z_med; 
                    break; 
                } else {
                    window_size += 2;
                    if (window_size > S_MAX) {
                        result_color = Z_med; 
                        break;
                    }
                }
            }
            // Salvataggio del risultato finale
            out_data[(gy * width + gx) * channels + c] = result_color;
        }

        // ========================================================
        // FASE 3: SINCRONIZZAZIONE DI FINE CANALE
        // ========================================================
        // FONDAMENTALE! Dobbiamo aspettare che tutti abbiano finito di leggere
        // s_in prima di ricaricarlo da capo per il prossimo canale colore!
        __syncthreads();
    }
}

// ... [INSERISCI QUI LA TUA FUNZIONE load_image E save_image] ...

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <input.ppm> [output.ppm]\n", argv[0]);
        return 1;
    }
    
    const char *input_file = argv[1];
    const char *output_file = (argc > 2) ? argv[2] : "output_shared.ppm";
    
    int width, height, channels;
    unsigned char *h_img_in = load_image(input_file, &width, &height, &channels);
    if (!h_img_in) return 1;
    
    size_t img_size = width * height * channels * sizeof(unsigned char);
    unsigned char *h_img_out = (unsigned char*)malloc(img_size);
    
    unsigned char *d_img_in, *d_img_out;
    cudaCheckError(cudaMalloc((void**)&d_img_in, img_size));
    cudaCheckError(cudaMalloc((void**)&d_img_out, img_size));
    cudaCheckError(cudaMemcpy(d_img_in, h_img_in, img_size, cudaMemcpyHostToDevice));
    
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 numBlocks((width + TILE_SIZE - 1) / TILE_SIZE,
                   (height + TILE_SIZE - 1) / TILE_SIZE);

    printf("Avvio AMF GPU (SHARED MEMORY)...\n");

    // --- SETUP CRONOMETRO GPU ---
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start); // Fai partire il timer
    
    // Lancio Kernel
    adaptive_median_shared_kernel<<<numBlocks, threadsPerBlock>>>(d_img_in, d_img_out, width, height, channels);
    
    cudaEventRecord(stop); // Ferma il timer
    // ----------------------------

    cudaCheckError(cudaPeekAtLastError());
    cudaCheckError(cudaDeviceSynchronize()); // Aspetta fine lavoro
    
    // Calcola e stampa tempo
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Tempo di esecuzione GPU: %.3f ms\n", milliseconds);
    
    cudaCheckError(cudaMemcpy(h_img_out, d_img_out, img_size, cudaMemcpyDeviceToHost));
    
    // save_image(output_file, h_img_out, width, height, channels);
    printf("Completato.\n");
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_img_in);
    cudaFree(d_img_out);
    free(h_img_in);
    free(h_img_out);
    
    return 0;
}