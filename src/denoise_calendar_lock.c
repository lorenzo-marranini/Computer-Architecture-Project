#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>
#include <stdatomic.h>
//#include <AMDProfileController.h>

const int S_MAX = 15;

typedef struct {
    unsigned char *in_data;
    unsigned char *out_data;
    int width;
    int height;
    int channels;
    int chunk_size;
    atomic_int *global_row_counter; // Keeps the dynamic load balancing
    
    // Stats to prove it's actually working
} ThreadData;

void* adaptive_median_worker(void* arg) {
    ThreadData* td = (ThreadData*)arg;
    int buckets[256];
    
    while (1) {
        int start_row = atomic_fetch_add(td->global_row_counter, td->chunk_size);
        if (start_row >= td->height) {
            break;
        }
        
        int end_row = start_row + td->chunk_size;
        if (end_row > td->height) {
            end_row = td->height;
        }

        for (int y = start_row; y < end_row; y++) {
            for (int x = 0; x < td->width; x++) {
                for (int c = 0; c < td->channels; c++) {
                    
                    int out_idx = (y * td->width + x) * td->channels + c;
                    int Z_xy = td->in_data[out_idx];
                    
                    int window_size = 3;
                    int result_color = Z_xy;

                    while (window_size <= S_MAX) {
                        // Clear the bucket and read every single pixel in the window 
                        // from scratch, every time we expand.
                        memset(buckets, 0, sizeof(buckets));
                        
                        int r = window_size / 2;
                        for (int wy = -r; wy <= r; wy++) {
                            for (int wx = -r; wx <= r; wx++) {
                                
                                int ny = y + wy;
                                int nx = x + wx;
                                
                                if (ny < 0) ny = 0;
                                if (ny >= td->height) ny = td->height - 1;
                                if (nx < 0) nx = 0;
                                if (nx >= td->width) nx = td->width - 1;
                                
                                int n_idx = (ny * td->width + nx) * td->channels + c;
                                buckets[td->in_data[n_idx]]++;
                            }
                        }

                        // Calculate median from the freshly populated buckets
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
    }
    return NULL;
}

unsigned char* load_image(const char *filename, int *w, int *h, int *channels) {
    FILE *f = fopen(filename, "rb");
    if (!f) return NULL;
    
    char magic[3];
    fscanf(f, "%2s", magic);
    
    if (strcmp(magic, "P5") == 0) {
        *channels = 1;
    } else if (strcmp(magic, "P6") == 0) {
        *channels = 3;
    } else {
        fclose(f);
        return NULL;
    }
    
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
    
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <input.ppm> [output.ppm] [threads]\n", argv[0]);
        return 1;
    }
    
    const char *input_file = argv[1];
    int num_threads = (argc > 3) ? atoi(argv[3]) : 6;
    
    int width, height, channels;
    unsigned char *img_in = load_image(input_file, &width, &height, &channels);
    if (!img_in) {
        fprintf(stderr, "Errore caricamento immagine\n");
        return 1;
    }
    const char *output_file = (argc > 2) ? argv[2] : "output.ppm";

    unsigned char *img_out = malloc(width * height * channels);
    
    atomic_int current_global_row = 0;
    int chunk_size = 16; 
    
    pthread_t threads[num_threads];
    ThreadData td[num_threads];

    printf("Filtro Mediano Adattivo: Naive Bucket + Dynamic Sched (%d thread, chunk=%d)...\n", num_threads, chunk_size);
    struct timespec start_time, end_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);
    //amdProfileResume();

    for (int i = 0; i < num_threads; i++) {
        td[i].in_data = img_in;
        td[i].out_data = img_out;
        td[i].width = width;
        td[i].height = height;
        td[i].channels = channels;
        td[i].chunk_size = chunk_size;
        td[i].global_row_counter = &current_global_row;
        
        pthread_create(&threads[i], NULL, adaptive_median_worker, &td[i]);
    }

    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }
    clock_gettime(CLOCK_MONOTONIC, &end_time);
    
    // Calculate elapsed time in seconds
    double elapsed = (end_time.tv_sec - start_time.tv_sec) + 
                     (end_time.tv_nsec - start_time.tv_nsec) / 1e9;

    // We print a specific tag "COMPUTE_TIME:" so Python can find it easily
    
    //amdProfilePause();
    printf("COMPUTE_TIME: %f\n", elapsed);
    save_image(output_file, img_out, width, height, channels);
    
    
    free(img_in);
    free(img_out);
    
    return 0;
}