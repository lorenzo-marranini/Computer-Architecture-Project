#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>
#include <AMDProfileController.h>

// Dimensione massima della finestra per il filtro adattivo (deve essere dispari).
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

// Il Worker che utilizza la logica Calendar Sort (Counting Sort)
void* adaptive_median_worker(void* arg) {
    ThreadData* td = (ThreadData*)arg;
    int buckets[256];
    
    for (int y = td->start_row; y < td->end_row; y++) {
        for (int x = 0; x < td->width; x++) {
            for (int c = 0; c < td->channels; c++) {
                
                int out_idx = (y * td->width + x) * td->channels + c;
                int Z_xy = td->in_data[out_idx];
                
                // RESET: I bucket vengono puliti solo quando passiamo a un NUOVO pixel
                memset(buckets, 0, sizeof(buckets));
                
                int window_size = 3;
                int result_color = Z_xy;
                int current_r = 0; // Raggio della finestra già elaborata

                while (window_size <= S_MAX) {
                    int target_r = window_size / 2;
                    
                    // AGGIUNTA INCREMENTALE: 
                    // Aggiungiamo solo i pixel che stanno tra current_r e target_r
                    for (int wy = -target_r; wy <= target_r; wy++) {
                        for (int wx = -target_r; wx <= target_r; wx++) {
                            
                            // Un pixel fa parte della "nuova cornice" se almeno una 
                            // delle sue coordinate relative è uguale al raggio target
                            if (abs(wy) == target_r || abs(wx) == target_r) {
                                int ny = y + wy;
                                int nx = x + wx;
                                
                                // Clamping
                                if (ny < 0) ny = 0;
                                if (ny >= td->height) ny = td->height - 1;
                                if (nx < 0) nx = 0;
                                if (nx >= td->width) nx = td->width - 1;
                                
                                int n_idx = (ny * td->width + nx) * td->channels + c;
                                buckets[td->in_data[n_idx]]++;
                            }
                        }
                    }
                    
                    // Aggiorniamo il raggio corrente: questi pixel sono ora nei bucket
                    current_r = target_r;

                    // Calcolo Min, Max e Mediana (stessa logica di prima)
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
                    
                    // Logica Adaptive
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

    printf("Applicazione FILTRO MEDIANO ADATTIVO (Calendar Sort) con %d thread...\n", num_threads);

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