#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>

// Standard 4x4 Bayer Matrix
const int BAYER_SIZE = 4;
const int BAYER_MATRIX[4][4] = {
    { 0,  8,  2, 10},
    {12,  4, 14,  6},
    { 3, 11,  1,  9},
    {15,  7, 13,  5}
};

typedef struct {
    unsigned char *data;
    int width;
    int height;
    int start_row;
    int end_row;
    int channels; // 1 for grayscale, 3 for RGB
} ThreadData;

// The core dithering logic for grayscale
void* dither_section(void* arg) {
    ThreadData* td = (ThreadData*)arg;
    
    for (int y = td->start_row; y < td->end_row; y++) {
        for (int x = 0; x < td->width; x++) {
            int index = y * td->width + x;
            int threshold = BAYER_MATRIX[y % BAYER_SIZE][x % BAYER_SIZE];
            int pixel_val = td->data[index] / 16;
            td->data[index] = (pixel_val > threshold) ? 255 : 0;
        }
    }
    return NULL;
}

// The core dithering logic for RGB
void* dither_section_rgb(void* arg) {
    ThreadData* td = (ThreadData*)arg;
    
    for (int y = td->start_row; y < td->end_row; y++) {
        for (int x = 0; x < td->width; x++) {
            int index = (y * td->width + x) * 3; // 3 channels per pixel
            int threshold = BAYER_MATRIX[y % BAYER_SIZE][x % BAYER_SIZE];
            
            for (int c = 0; c < 3; c++) {
                int pixel_val = td->data[index + c] / 16;
                td->data[index + c] = (pixel_val > threshold) ? 255 : 0;
            }
        }
    }
    return NULL;
}

void ordered_dither_mt(unsigned char* image, int w, int h, int num_threads, int channels) {
    pthread_t threads[num_threads];
    ThreadData td[num_threads];
    
    int rows_per_thread = h / num_threads;
    void* (*dither_func)(void*) = (channels == 1) ? dither_section : dither_section_rgb;

    for (int i = 0; i < num_threads; i++) {
        td[i].data = image;
        td[i].width = w;
        td[i].height = h;
        td[i].channels = channels;
        td[i].start_row = i * rows_per_thread;
        td[i].end_row = (i == num_threads - 1) ? h : (i + 1) * rows_per_thread;

        pthread_create(&threads[i], NULL, dither_func, &td[i]);
    }

    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }
}

// Load a PGM or PPM image from file
unsigned char* load_pgm(const char *filename, int *w, int *h, int *channels) {
    FILE *f = fopen(filename, "rb");
    if (!f) {
        fprintf(stderr, "Error: Could not open file '%s'\n", filename);
        return NULL;
    }
    
    printf("Cariamento immagine...\n");

    char magic[3];
    fscanf(f, "%2s", magic);
    
    if (strcmp(magic, "P5") == 0) {
        *channels = 1; // Grayscale
    } else if (strcmp(magic, "P6") == 0) {
        *channels = 3; // RGB
    } else {
        fprintf(stderr, "Error: File is not a valid PGM (P5) or PPM (P6) image\n");
        fclose(f);
        return NULL;
    }

    printf("Channels read: %d \n", *channels);


    int max_val;
    fscanf(f, "%d %d %d", w, h, &max_val);
    fgetc(f); // consume whitespace
    
    unsigned char *data = malloc(*w * *h * *channels);
    if (!data) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        fclose(f);
        return NULL;
    }
    
    size_t read = fread(data, 1, *w * *h * *channels, f);
    if (read != (size_t)(*w * *h * *channels)) {
        fprintf(stderr, "Error: Could not read image data\n");
        free(data);
        fclose(f);
        return NULL;
    }
    
    fclose(f);
    return data;
}

void save_pgm(const char *filename, unsigned char *data, int w, int h, int channels) {
    FILE *f = fopen(filename, "wb");
    if (!f) return;
    
    const char *magic = (channels == 1) ? "P5" : "P6";
    fprintf(f, "%s\n%d %d\n255\n", magic, w, h);
    fwrite(data, 1, w * h * channels, f);
    fclose(f);
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <input_image.pgm|.ppm> [output_image.pgm|.ppm] [num_threads]\n", argv[0]);
        fprintf(stderr, "Example: %s input.pgm output.pgm 4\n", argv[0]);
        return 1;
    }
    
    const char *input_file = argv[1];
    const char *output_file = (argc > 2) ? argv[2] : "output.pgm";
    int threads = (argc > 3) ? atoi(argv[3]) : 4;
    
    int width, height, channels;
    unsigned char *img = load_pgm(input_file, &width, &height, &channels);

    
    if (!img) {
        return 1;
    }
    
    const char *format = (channels == 1) ? "grayscale" : "RGB";
    printf("Loaded image: %dx%d (%s)\n", width, height, format);
    printf("Processing with %d threads...\n", threads);
    ordered_dither_mt(img, width, height, threads, channels);
    save_pgm(output_file, img, width, height, channels);
    printf("Saved to: %s\n", output_file);
    free(img);
    return 0;
}