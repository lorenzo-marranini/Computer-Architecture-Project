#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>
#include <AMDProfileController.h>

const int S_MAX = 15; 

typedef struct {
    unsigned char *in_data;
    unsigned char *out_data;
    int width;
    int height;
    int channels;
    int start_row;
    int end_row;
} ThreadData;

// MACRO per evitare di riscrivere il clamping e il calcolo indice 5 volte.
// Usare una macro garantisce che non ci sia overhead di chiamata a funzione.
#define ADD_PIXEL(wy, wx) \
    do { \
        int ny = y + (wy); \
        int nx = x + (wx); \
        if (ny < 0) ny = 0; \
        if (ny >= td->height) ny = td->height - 1; \
        if (nx < 0) nx = 0; \
        if (nx >= td->width) nx = td->width - 1; \
        int n_idx = (ny * td->width + nx) * td->channels + c; \
        buckets[td->in_data[n_idx]]++; \
    } while(0)

void* adaptive_median_worker(void* arg) {
    ThreadData* td = (ThreadData*)arg;
    int buckets[256];
    
    for (int y = td->start_row; y < td->end_row; y++) {
        for (int x = 0; x < td->width; x++) {
            for (int c = 0; c < td->channels; c++) {
                
                int out_idx = (y * td->width + x) * td->channels + c;
                int Z_xy = td->in_data[out_idx];
                
                memset(buckets, 0, sizeof(buckets));
                
                int window_size = 3;
                int result_color = Z_xy;

                while (window_size <= S_MAX) {
                    
                    if (window_size == 3) {
                        // CASO BASE 3x3: Riempiamo tutto il blocco centrale (9 pixel)
                        for (int wy = -1; wy <= 1; wy++) {
                            for (int wx = -1; wx <= 1; wx++) {
                                ADD_PIXEL(wy, wx);
                            }
                        }
                    } else {
                        // CASO ESPANSIONE (>3): Aggiungiamo SOLO la cornice esterna
                        int r = window_size / 2;
                        
                        // 1 & 2. RIGA SUPERIORE E RIGA INFERIORE (da angolo ad angolo)
                        for (int wx = -r; wx <= r; wx++) {
                            ADD_PIXEL(-r, wx); // Riga Top (wy = -r)
                            ADD_PIXEL(r, wx);  // Riga Bottom (wy = r)
                        }
                        
                        // 3 & 4. COLONNA SINISTRA E COLONNA DESTRA
                        // (escludiamo gli angoli estremi wy = -r e wy = r perché già letti sopra)
                        for (int wy = -r + 1; wy <= r - 1; wy++) {
                            ADD_PIXEL(wy, -r); // Colonna Left (wx = -r)
                            ADD_PIXEL(wy, r);  // Colonna Right (wx = r)
                        }
                    }

                    // Calcolo statistiche sui bucket
                    int Z_min = -1, Z_max = -1, Z_med = -1;
                    int count_cumulativo = 0;
                    int total_elements = window_size * window_size;
                    int target_pos = (total_elements / 2) + 1; 
                    
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
                    
                    // Logica Adaptive (Livelli A e B)
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
                td->out_data[out_idx] = result_color;
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
    amdProfileResume();
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
    pthread_t threads[num_threads];
    ThreadData td[num_threads];
    int rows_per_thread = height / num_threads;

    printf("Applicazione FILTRO MEDIANO ADATTIVO (Calendar Sort No-If) con %d thread...\n", num_threads);

    for (int i = 0; i < num_threads; i++) {
        td[i].in_data = img_in;
        td[i].out_data = img_out;
        td[i].width = width;
        td[i].height = height;
        td[i].channels = channels;
        td[i].start_row = i * rows_per_thread;
        td[i].end_row = (i == num_threads - 1) ? height : (i + 1) * rows_per_thread;
        pthread_create(&threads[i], NULL, adaptive_median_worker, &td[i]);
    }

    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }

    save_image(output_file, img_out, width, height, channels);
    printf("Completato. Salvato in %s\n", output_file);
    
    free(img_in);
    free(img_out);
    return 0;
}