#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>
#include <stdatomic.h>
#include <AMDProfileController.h>

const int S_MAX = 15; 

// MACRO per gestire i bordi in modo pulito ed efficiente
#define CLAMP(val, min, max) ((val) < (min) ? (min) : ((val) > (max) ? (max) : (val)))

typedef struct {
    unsigned char *in_data;
    unsigned char *out_data;
    int width;
    int height;
    int channels;
    int chunk_size;
    atomic_int *global_row_counter; 
    
    long long count_5x5;
    long long count_7x7;
    long long count_9x9;
    long long count_larger;
} ThreadData;

void* adaptive_median_worker(void* arg) {
    ThreadData* td = (ThreadData*)arg;
    
    while (1) {
        int start_row = atomic_fetch_add(td->global_row_counter, td->chunk_size);
        if (start_row >= td->height) break;
        
        int end_row = start_row + td->chunk_size;
        if (end_row > td->height) end_row = td->height;

        for (int y = start_row; y < end_row; y++) {
            for (int c = 0; c < td->channels; c++) {
                
                // L'ISTOGRAMMA BASE: Contiene sempre e solo il 5x5 corrente
                int base_buckets[256];
                memset(base_buckets, 0, sizeof(base_buckets));
                
                // 1. Inizializza il base_buckets SOLO per il primo pixel (x = 0)
                for (int wy = -2; wy <= 2; wy++) {
                    int ny = CLAMP(y + wy, 0, td->height - 1);
                    for (int wx = -2; wx <= 2; wx++) {
                        int nx = CLAMP(0 + wx, 0, td->width - 1);
                        int n_idx = (ny * td->width + nx) * td->channels + c;
                        base_buckets[td->in_data[n_idx]]++;
                    }
                }

                // 2. Scorriamo la riga spostando la finestra (HUANG'S ALGORITHM)
                for (int x = 0; x < td->width; x++) {
                    int out_idx = (y * td->width + x) * td->channels + c;
                    int Z_xy = td->in_data[out_idx];
                    
                    // SLIDING LOGIC: Aggiorna l'istogramma base per i pixel successivi
                    if (x > 0) {
                        // Rimuovi la colonna di sinistra che esce (x - 3)
                        int left_x = CLAMP(x - 3, 0, td->width - 1);
                        for (int wy = -2; wy <= 2; wy++) {
                            int ny = CLAMP(y + wy, 0, td->height - 1);
                            int n_idx = (ny * td->width + left_x) * td->channels + c;
                            base_buckets[td->in_data[n_idx]]--;
                        }
                        
                        // Aggiungi la colonna di destra che entra (x + 2)
                        int right_x = CLAMP(x + 2, 0, td->width - 1);
                        for (int wy = -2; wy <= 2; wy++) {
                            int ny = CLAMP(y + wy, 0, td->height - 1);
                            int n_idx = (ny * td->width + right_x) * td->channels + c;
                            base_buckets[td->in_data[n_idx]]++;
                        }
                    }
                    
                    // Copiamo il 5x5 base in un istogramma temporaneo.
                    // Se dobbiamo espandere a 7x7 o 9x9, "sporchiamo" solo questa copia!
                    int buckets[256];
                    memcpy(buckets, base_buckets, sizeof(buckets));
                    
                    int window_size = 5;
                    int result_color = Z_xy;

                    while (window_size <= S_MAX) {
                        
                        // Se window_size è > 5, aggiungiamo il perimetro al working bucket
                        if (window_size > 5) {
                            int r = window_size / 2;
                            
                            // Bordi Orizzontali (Top e Bottom)
                            int ny_top = CLAMP(y - r, 0, td->height - 1);
                            int ny_bot = CLAMP(y + r, 0, td->height - 1);
                            for (int wx = -r; wx <= r; wx++) {
                                int nx = CLAMP(x + wx, 0, td->width - 1);
                                buckets[td->in_data[(ny_top * td->width + nx) * td->channels + c]]++;
                                buckets[td->in_data[(ny_bot * td->width + nx) * td->channels + c]]++;
                            }
                            // Bordi Verticali (Left e Right, escludendo gli angoli già contati)
                            int nx_left = CLAMP(x - r, 0, td->width - 1);
                            int nx_right = CLAMP(x + r, 0, td->width - 1);
                            for (int wy = -r + 1; wy <= r - 1; wy++) {
                                int ny = CLAMP(y + wy, 0, td->height - 1);
                                buckets[td->in_data[(ny * td->width + nx_left) * td->channels + c]]++;
                                buckets[td->in_data[(ny * td->width + nx_right) * td->channels + c]]++;
                            }
                        }

                        // Trova Z_min, Z_max, e Z_med
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
                        
                        // Livello A e Livello B
                        if ((Z_med - Z_min) > 0 && (Z_med - Z_max) < 0) {
                            if ((Z_xy - Z_min) > 0 && (Z_xy - Z_max) < 0) {
                                result_color = Z_xy; 
                            } else {
                                result_color = Z_med; 
                            }
                            
                            if (window_size == 5) td->count_5x5++;
                            else if (window_size == 7) td->count_7x7++;
                            else if (window_size == 9) td->count_9x9++;
                            else td->count_larger++;
                            
                            break; 
                        } else {
                            window_size += 2;
                            if (window_size > S_MAX) {
                                result_color = Z_med; 
                                
                                if (S_MAX == 5) td->count_5x5++;
                                else if (S_MAX == 7) td->count_7x7++;
                                else if (S_MAX == 9) td->count_9x9++;
                                else td->count_larger++;
                                
                                break;
                            }
                        }
                    }
                    td->out_data[out_idx] = result_color;
                }
            }
        }
    }
    return NULL;
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
    unsigned char *data = malloc(*w * *h * *channels);
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
    amdProfileResume(); // START UPROF
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <input.ppm> [output.ppm] [threads]\n", argv[0]);
        return 1;
    }
    
    const char *input_file = argv[1];
    const char *output_file = (argc > 2) ? argv[2] : "output.ppm";
    int num_threads = (argc > 3) ? atoi(argv[3]) : 6;
    
    int width, height, channels;
    unsigned char *img_in = load_image(input_file, &width, &height, &channels);
    if (!img_in) {
        fprintf(stderr, "Errore caricamento immagine\n");
        return 1;
    }
    
    unsigned char *img_out = malloc(width * height * channels);
    
    atomic_int current_global_row = 0;
    int chunk_size = 32; 
    
    pthread_t threads[num_threads];
    ThreadData td[num_threads];

    printf("Filtro Mediano Adattivo: True Sliding Window (%d thread, chunk=%d)...\n", num_threads, chunk_size);

    for (int i = 0; i < num_threads; i++) {
        td[i].in_data = img_in;
        td[i].out_data = img_out;
        td[i].width = width;
        td[i].height = height;
        td[i].channels = channels;
        td[i].chunk_size = chunk_size;
        td[i].global_row_counter = &current_global_row;
        
        td[i].count_5x5 = 0;
        td[i].count_7x7 = 0;
        td[i].count_9x9 = 0;
        td[i].count_larger = 0;
        
        pthread_create(&threads[i], NULL, adaptive_median_worker, &td[i]);
    }

    long long total_5x5 = 0, total_7x7 = 0, total_9x9 = 0, total_larger = 0;

    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
        
        total_5x5 += td[i].count_5x5;
        total_7x7 += td[i].count_7x7;
        total_9x9 += td[i].count_9x9;
        total_larger += td[i].count_larger;
    }

    //save_image(output_file, img_out, width, height, channels);
    
    long long total_pixels = total_5x5 + total_7x7 + total_9x9 + total_larger;
    printf("\n--- STATISTICHE ADAPTIVE MEDIAN FILTER ---\n");
    printf("Canali elaborati totali : %lld\n", total_pixels);
    printf("Blocchi 5x5  utilizzati : %lld (%.2f%%)\n", total_5x5, (double)total_5x5 / total_pixels * 100.0);
    printf("Blocchi 7x7  utilizzati : %lld (%.2f%%)\n", total_7x7, (double)total_7x7 / total_pixels * 100.0);
    printf("Blocchi 9x9  utilizzati : %lld (%.2f%%)\n", total_9x9, (double)total_9x9 / total_pixels * 100.0);
    printf("Blocchi >9x9 utilizzati : %lld (%.2f%%)\n", total_larger, (double)total_larger / total_pixels * 100.0);
    printf("------------------------------------------\n\n");
    
    printf("Completato. Salvato in %s\n", output_file);
    
    amdProfilePause(); // STOP UPROF
    
    free(img_in);
    free(img_out);
    return 0;
}