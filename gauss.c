#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>

typedef struct {
    unsigned char *in_data;
    unsigned char *out_data;
    int width;
    int height;
    int channels;
    int start_row;
    int end_row;
} ThreadData;

// The BLUR_RADIUS controls how heavy the algorithm is.
// Radius 20 = (20*2+1)^2 = 1681 pixels checked per 1 output pixel.
const int BLUR_RADIUS = 20; 

// The Heavy Convolution Worker (Embarrassingly Parallel)
void* heavy_blur_worker(void* arg) {
    ThreadData* td = (ThreadData*)arg;
    
    for (int y = td->start_row; y < td->end_row; y++) {
        for (int x = 0; x < td->width; x++) {
            
            long r_total = 0, g_total = 0, b_total = 0;
            int count = 0;
            
            // Look at every pixel in the BLUR_RADIUS around our target
            for (int ky = -BLUR_RADIUS; ky <= BLUR_RADIUS; ky++) {
                for (int kx = -BLUR_RADIUS; kx <= BLUR_RADIUS; kx++) {
                    
                    int neighbor_y = y + ky;
                    int neighbor_x = x + kx;
                    
                    // Make sure we don't read outside the image boundaries
                    if (neighbor_y >= 0 && neighbor_y < td->height && 
                        neighbor_x >= 0 && neighbor_x < td->width) {
                        
                        int idx = (neighbor_y * td->width + neighbor_x) * td->channels;
                        
                        if (td->channels == 3) {
                            r_total += td->in_data[idx];
                            g_total += td->in_data[idx + 1];
                            b_total += td->in_data[idx + 2];
                        } else {
                            r_total += td->in_data[idx]; // Grayscale
                        }
                        count++;
                    }
                }
            }
            
            // Average the colors and write to the output image
            int out_idx = (y * td->width + x) * td->channels;
            if (td->channels == 3) {
                td->out_data[out_idx] = r_total / count;
                td->out_data[out_idx + 1] = g_total / count;
                td->out_data[out_idx + 2] = b_total / count;
            } else {
                td->out_data[out_idx] = r_total / count;
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
    const char *output_file = (argc > 2) ? argv[2] : "output.ppm";
    int num_threads = (argc > 3) ? atoi(argv[3]) : 8;
    
    int width, height, channels;
    unsigned char *img_in = load_image(input_file, &width, &height, &channels);
    if (!img_in) {
        printf("Error loading image!\n");
        return 1; 
    }
    
    unsigned char *img_out = malloc(width * height * channels);
    pthread_t threads[num_threads];
    ThreadData td[num_threads];
    int rows_per_thread = height / num_threads;

    printf("Loaded %dx%d (%d channels)\n", width, height, channels);
    printf("Applying Heavy Blur (Radius %d) with %d threads...\n", BLUR_RADIUS, num_threads);

    for (int i = 0; i < num_threads; i++) {
        td[i].in_data = img_in;
        td[i].out_data = img_out;
        td[i].width = width;
        td[i].height = height;
        td[i].channels = channels;
        td[i].start_row = i * rows_per_thread;
        td[i].end_row = (i == num_threads - 1) ? height : (i + 1) * rows_per_thread;
        
        pthread_create(&threads[i], NULL, heavy_blur_worker, &td[i]);
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