#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>
#include <math.h>
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

// Estrae min, max e mediana dall'istogramma
void get_stats_from_hist(int hist[256], int total_pixels, int *min, int *max, int *med) {
    int count = 0;
    int found_min = -1;
    *med = -1; // Inizializzazione di sicurezza
    *min = 0;
    *max = 255;

    for (int i = 0; i < 256; i++) {
        if (hist[i] > 0) {
            if (found_min == -1) found_min = i;
            *max = i;
            count += hist[i];
            if (count >= (total_pixels / 2) + 1 && *med == -1) {
                *med = i;
            }
        }
    }
    *min = (found_min == -1) ? 0 : found_min;
    // Se non troviamo la mediana (istogramma vuoto, non dovrebbe succedere), mettiamo 128
    if (*med == -1) *med = 128; 
}

// Riempie l'istogramma da zero per una data finestra
void fill_histogram(int hist[256], ThreadData* td, int x, int y, int c, int size) {
    memset(hist, 0, 256 * sizeof(int));
    int half = size / 2;
    for (int wy = -half; wy <= half; wy++) {
        for (int wx = -half; wx <= half; wx++) {
            int ny = (int)fmax(0, fmin(td->height - 1, y + wy));
            int nx = (int)fmax(0, fmin(td->width - 1, x + wx));
            hist[td->in_data[(ny * td->width + nx) * td->channels + c]]++;
        }
    }
}

void* adaptive_median_worker(void* arg) {
    ThreadData* td = (ThreadData*)arg;
    int hist[256];
    int needs_full_rebuild = 1; 

    for (int y = td->start_row; y < td->end_row; y++) {
        for (int c = 0; c < td->channels; c++) {
            needs_full_rebuild = 1; 

            for (int x = 0; x < td->width; x++) {
                int current_idx = (y * td->width + x) * td->channels + c;
                int z_xy = td->in_data[current_idx];
                
                // 1. GESTIONE ISTOGRAMMA BASE (3x3)
                if (needs_full_rebuild) {
                    fill_histogram(hist, td, x, y, c, 3);
                    needs_full_rebuild = 0;
                }

                int win_size = 3;
                int z_min = 0, z_max = 255, z_med = 128; // Inizializzazione per evitare warning
                int final_pixel = z_xy;

                // 2. LOGICA ADAPTIVE (INCREMENTALE)
                while (win_size <= S_MAX) {
                    get_stats_from_hist(hist, win_size * win_size, &z_min, &z_max, &z_med);

                    if (z_med > z_min && z_med < z_max) { // Level A
                        if (z_xy > z_min && z_xy < z_max) final_pixel = z_xy; // Level B
                        else final_pixel = z_med;
                        break;
                    } else {
                        // Allargando, l'istogramma 3x3 viene "sporcato"
                        needs_full_rebuild = 1; 
                        
                        int old_half = win_size / 2;
                        win_size += 2;
                        if (win_size > S_MAX) {
                            final_pixel = z_med;
                            break;
                        }
                        int new_half = win_size / 2;

                        // AGGIUNTA INCREMENTALE: Solo la nuova cornice esterna
                        for (int i = -new_half; i <= new_half; i++) {
                            for (int j = -new_half; j <= new_half; j++) {
                                if (abs(i) > old_half || abs(j) > old_half) {
                                    int ny = (int)fmax(0, fmin(td->height - 1, y + i));
                                    int nx = (int)fmax(0, fmin(td->width - 1, x + j));
                                    hist[td->in_data[(ny * td->width + nx) * td->channels + c]]++;
                                }
                            }
                        }
                    }
                }
                td->out_data[current_idx] = (unsigned char)final_pixel;

                // 3. SLIDING WINDOW (Solo se l'istogramma è rimasto 3x3)
                if (!needs_full_rebuild && x < td->width - 1) {
                    for (int wy = -1; wy <= 1; wy++) {
                        int ny = (int)fmax(0, fmin(td->height - 1, y + wy));
                        
                        int col_out_idx = (int)fmax(0, fmin(td->width - 1, x - 1));
                        int col_in_idx  = (int)fmax(0, fmin(td->width - 1, x + 2));
                        
                        int val_out = td->in_data[(ny * td->width + col_out_idx) * td->channels + c];
                        int val_in  = td->in_data[(ny * td->width + col_in_idx) * td->channels + c];
                        
                        hist[val_out]--;
                        hist[val_in]++;
                    }
                }
            }
        }
    }
    return NULL;
}

// Caricamento immagine (PPM P5/P6)
unsigned char* load_image(const char *filename, int *w, int *h, int *channels) {
    FILE *f = fopen(filename, "rb");
    if (!f) return NULL;
    char magic[3];
    if (fscanf(f, "%2s", magic) != 1) { fclose(f); return NULL; }
    if (strcmp(magic, "P5") == 0) *channels = 1;
    else if (strcmp(magic, "P6") == 0) *channels = 3;
    else { fclose(f); return NULL; }
    int max_val;
    if (fscanf(f, "%d %d %d", w, h, &max_val) != 3) { fclose(f); return NULL; }
    fgetc(f); 
    unsigned char *data = malloc(*w * *h * *channels);
    if (data) fread(data, 1, *w * *h * *channels, f);
    fclose(f);
    return data;
}

// Salvataggio immagine
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
        fprintf(stderr, "Error loading image %s\n", input_file);
        return 1;
    }
    
    unsigned char *img_out = malloc(width * height * channels);
    pthread_t threads[num_threads];
    ThreadData td[num_threads];
    int rows_per_thread = height / num_threads;

    printf("Applying ADAPTIVE MEDIAN FILTER with %d threads...\n", num_threads);

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
    printf("Done. Saved to %s\n", output_file);
    
    free(img_in);
    free(img_out);
    return 0;
}