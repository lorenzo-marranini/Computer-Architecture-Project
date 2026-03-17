#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>

// Maximum window size for the adaptive filter (must be an odd number).
// A larger S_MAX means a heavier potential workload for noisy pixels.
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

// Helper function for the C standard library quicksort
int compare_ints(const void* a, const void* b) {
    return (*(int*)a - *(int*)b);
}

// The Academic Adaptive Median Filter Worker
void* adaptive_median_worker(void* arg) {
    ThreadData* td = (ThreadData*)arg;
    
    // Allocate a buffer large enough for the maximum possible window size
    int max_elements = S_MAX * S_MAX;
    int* window = (int*)malloc(max_elements * sizeof(int));
    
    for (int y = td->start_row; y < td->end_row; y++) {
        for (int x = 0; x < td->width; x++) {
            for (int c = 0; c < td->channels; c++) { // Process RGB independently
                
                int out_idx = (y * td->width + x) * td->channels + c;
                int Z_xy = td->in_data[out_idx]; // The current pixel color
                
                int window_size = 3; // Always start with a 3x3 window
                int result_color = Z_xy;
                
                // DATA-DEPENDENT LOOP: This is what causes the load imbalance!
                while (window_size <= S_MAX) {
                    int half_w = window_size / 2;
                    int count = 0;
                    
                    // Extract the pixels in the current window size
                    for (int wy = -half_w; wy <= half_w; wy++) {
                        for (int wx = -half_w; wx <= half_w; wx++) {
                            int ny = y + wy;
                            int nx = x + wx;
                            
                            // Clamp to image boundaries
                            if (ny < 0) ny = 0;
                            if (ny >= td->height) ny = td->height - 1;
                            if (nx < 0) nx = 0;
                            if (nx >= td->width) nx = td->width - 1;
                            
                            int n_idx = (ny * td->width + nx) * td->channels + c;
                            window[count++] = td->in_data[n_idx];
                        }
                    }
                    
                    // Sort the array to find min, max, and median
                    qsort(window, count, sizeof(int), compare_ints);
                    
                    int Z_min = window[0];
                    int Z_max = window[count - 1];
                    int Z_med = window[count / 2];
                    
                    // Level A: Is the median itself a noisy pixel?
                    // (Assuming 0 is pepper noise, 255 is salt noise)
                    int A1 = Z_med - Z_min;
                    int A2 = Z_med - Z_max;
                    
                    if (A1 > 0 && A2 < 0) {
                        // Level B: The median is safe. Is the target pixel noisy?
                        int B1 = Z_xy - Z_min;
                        int B2 = Z_xy - Z_max;
                        
                        if (B1 > 0 && B2 < 0) {
                            result_color = Z_xy; // Pixel is fine, keep it
                        } else {
                            result_color = Z_med; // Pixel is noise, replace with median
                        }
                        break; // SUCCESS: Exit the while loop early!
                    } else {
                        // FAILURE: The median is noise too. 
                        // Increase the window size and try again.
                        window_size += 2;
                        
                        if (window_size > S_MAX) {
                            result_color = Z_med; // Give up and just use what we have
                            break;
                        }
                    }
                }
                
                td->out_data[out_idx] = result_color;
            }
        }
    }
    
    free(window);
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
    const char *output_file = (argc > 2) ? argv[2] : "output.ppm";
    int num_threads = (argc > 3) ? atoi(argv[3]) : 6;
    
    int width, height, channels;
    unsigned char *img_in = load_image(input_file, &width, &height, &channels);
    if (!img_in) return 1;
    
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