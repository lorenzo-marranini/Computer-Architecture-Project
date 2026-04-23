#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>
#include <stdatomic.h>
//#include <AMDProfileController.h>
#include <stdint.h>
#include <time.h>

const int S_MAX = 15; 

typedef struct {
    unsigned char *in_data;
    unsigned char *out_data;
    int width;
    int height;
    int channels;
    int chunk_size;
    atomic_int *global_row_counter; 
} ThreadData;


void* adaptive_median_worker(void* arg) {
    ThreadData* td = (ThreadData*)arg;
    uint8_t buckets[3][256];
    int max_radius = S_MAX / 2; 

    #define ADD_PIXEL_SAFE_RGB(wy, wx) \
        do { \
            int p_idx = ((y + (wy)) * td->width + (x + (wx))) * td->channels; \
            buckets[0][td->in_data[p_idx]]++; \
            buckets[1][td->in_data[p_idx + 1]]++; \
            buckets[2][td->in_data[p_idx + 2]]++; \
        } while(0)

    #define ADD_PIXEL_CLAMPED_RGB(wy, wx) \
        do { \
            int ny = y + (wy); \
            int nx = x + (wx); \
            if (ny < 0) ny = 0; else if (ny >= td->height) ny = td->height - 1; \
            if (nx < 0) nx = 0; else if (nx >= td->width) nx = td->width - 1; \
            int p_idx = (ny * td->width + nx) * td->channels; \
            buckets[0][td->in_data[p_idx]]++; \
            buckets[1][td->in_data[p_idx + 1]]++; \
            buckets[2][td->in_data[p_idx + 2]]++; \
        } while(0)

    while (1) {
        int start_row = atomic_fetch_add(td->global_row_counter, td->chunk_size);
        if (start_row >= td->height) break;
        
        int end_row = start_row + td->chunk_size;
        if (end_row > td->height) end_row = td->height;

        for (int y = start_row; y < end_row; y++) {
            for (int x = 0; x < td->width; x++) {
                
                int is_safe = (y >= max_radius && y < td->height - max_radius && 
                               x >= max_radius && x < td->width - max_radius);

                // Wipe all 3 channel histograms in one swift motion
                memset(buckets, 0, sizeof(buckets));
                
                // Track state for all 3 channels simultaneously
                int window_size = 3;
                int channels_done = 0; // Bitmask to track which channels are finished
                int result_color[3];
                
                // Base colors for Level A checks (Multiplication rolled back in)
                int base_idx = (y * td->width + x) * td->channels;
                int Z_xy[3] = { td->in_data[base_idx], td->in_data[base_idx+1], td->in_data[base_idx+2] };

                while (window_size <= S_MAX && channels_done != 7) { 
                    
                    int r = window_size / 2;
                    
                    if (is_safe) {
                        if (window_size == 3) {
                            for (int wy = -1; wy <= 1; wy++) {
                                for (int wx = -1; wx <= 1; wx++) ADD_PIXEL_SAFE_RGB(wy, wx);
                            }
                        } else {
                            for (int wx = -r; wx <= r; wx++) {
                                ADD_PIXEL_SAFE_RGB(-r, wx); 
                                ADD_PIXEL_SAFE_RGB(r, wx);  
                            }
                            for (int wy = -r + 1; wy <= r - 1; wy++) {
                                ADD_PIXEL_SAFE_RGB(wy, -r);
                                ADD_PIXEL_SAFE_RGB(wy, r);                                 
                            }
                        }
                    } else {
                        if (window_size == 3) {
                            for (int wy = -1; wy <= 1; wy++) {
                                for (int wx = -1; wx <= 1; wx++) ADD_PIXEL_CLAMPED_RGB(wy, wx);
                            }
                        } else {
                            for (int wx = -r; wx <= r; wx++) {
                                ADD_PIXEL_CLAMPED_RGB(-r, wx); 
                                ADD_PIXEL_CLAMPED_RGB(r, wx);  
                            }
                            for (int wy = -r + 1; wy <= r - 1; wy++) {
                                ADD_PIXEL_CLAMPED_RGB(wy, -r); 
                                ADD_PIXEL_CLAMPED_RGB(wy, r);  
                            }
                        }
                    }

                    // Process median logic for each channel independently
                    int target_pos = ((window_size * window_size) / 2) + 1; 

                    for (int c = 0; c < 3; c++) {
                        // Skip if this channel already found its median in a smaller window
                        if (channels_done & (1 << c)) continue; 

                        int Z_min = -1, Z_max = -1, Z_med = -1;
                        int count_cumulativo = 0;
                        
                        for (int i = 0; i < 256; i++) {
                            if (buckets[c][i] > 0) {
                                if (Z_min == -1) Z_min = i;
                                Z_max = i; 
                                count_cumulativo += buckets[c][i];
                                if (Z_med == -1 && count_cumulativo >= target_pos) {
                                    Z_med = i;
                                }
                            }
                        }
                        
                        // Adaptive Logic
                        if ((Z_med - Z_min) > 0 && (Z_med - Z_max) < 0) {
                            if ((Z_xy[c] - Z_min) > 0 && (Z_xy[c] - Z_max) < 0) {
                                result_color[c] = Z_xy[c]; 
                            } else {
                                result_color[c] = Z_med; 
                            }
                            channels_done |= (1 << c); // Mark this channel as completed
                        } else if (window_size + 2 > S_MAX) {
                            result_color[c] = Z_med; 
                            channels_done |= (1 << c); // Mark this channel as completed
                        }
                    }
                    window_size += 2;
                }
                
                // Write the completed colors back
                td->out_data[base_idx] = result_color[0];
                td->out_data[base_idx + 1] = result_color[1];
                td->out_data[base_idx + 2] = result_color[2];
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
    
    unsigned char *img_out = malloc(width * height * channels);
    const char *output_file = (argc > 2) ? argv[2] : "output.ppm";

    // The Atomic Load Balancer is active
    atomic_int current_global_row = 0;
    int chunk_size = 64; 
    
    pthread_t threads[num_threads];
    ThreadData td[num_threads];

    printf("Filtro Mediano Adattivo (%d thread, chunk=%d)...\n", num_threads, chunk_size);
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
    
    //save_image(output_file, img_out, width, height, channels);
    
    // Calculate elapsed time in seconds
    double elapsed = (end_time.tv_sec - start_time.tv_sec) + 
                     (end_time.tv_nsec - start_time.tv_nsec) / 1e9;

    // We print a specific tag "COMPUTE_TIME:" so Python can find it easily
    printf("COMPUTE_TIME: %f\n", elapsed);
    
    //amdProfilePause();
    
    free(img_in);
    free(img_out);
    return 0;
}