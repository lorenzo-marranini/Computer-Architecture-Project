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
} ThreadData;

// The core dithering logic
void* dither_section(void* arg) {
    ThreadData* td = (ThreadData*)arg;
    
    for (int y = td->start_row; y < td->end_row; y++) {
        for (int x = 0; x < td->width; x++) {
            int index = y * td->width + x;
            
            // Map 0-255 to 0-16 range for comparison with matrix
            int threshold = BAYER_MATRIX[y % BAYER_SIZE][x % BAYER_SIZE];
            int pixel_val = td->data[index] / 16; 

            // Basic binary thresholding
            td->data[index] = (pixel_val > threshold) ? 255 : 0;
        }
    }
    return NULL;
}

void ordered_dither_mt(unsigned char* image, int w, int h, int num_threads) {
    pthread_t threads[num_threads];
    ThreadData td[num_threads];
    
    int rows_per_thread = h / num_threads;

    for (int i = 0; i < num_threads; i++) {
        td[i].data = image;
        td[i].width = w;
        td[i].height = h;
        td[i].start_row = i * rows_per_thread;
        // Ensure the last thread covers any remaining rows
        td[i].end_row = (i == num_threads - 1) ? h : (i + 1) * rows_per_thread;

        pthread_create(&threads[i], NULL, dither_section, &td[i]);
    }

    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }
}

// Load a PGM image from file
unsigned char* load_pgm(const char *filename, int *w, int *h) {
    FILE *f = fopen(filename, "rb");
    if (!f) {
        fprintf(stderr, "Error: Could not open file '%s'\n", filename);
        return NULL;
    }
    
    char magic[3];
    fscanf(f, "%2s", magic);
    
    if (strcmp(magic, "P5") != 0) {
        fprintf(stderr, "Error: File is not a binary PGM image (P5 format)\n");
        fclose(f);
        return NULL;
    }
    
    int max_val;
    fscanf(f, "%d %d %d", w, h, &max_val);
    fgetc(f); // consume whitespace
    
    unsigned char *data = malloc(*w * *h);
    if (!data) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        fclose(f);
        return NULL;
    }
    
    size_t read = fread(data, 1, *w * *h, f);
    if (read != (size_t)(*w * *h)) {
        fprintf(stderr, "Error: Could not read image data\n");
        free(data);
        fclose(f);
        return NULL;
    }
    
    fclose(f);
    return data;
}

void save_pgm(const char *filename, unsigned char *data, int w, int h) {
    FILE *f = fopen(filename, "wb");
    if (!f) return;
    // P5 = Binary Grayscale, then Width, Height, and Max Value (255)
    fprintf(f, "P5\n%d %d\n255\n", w, h);
    fwrite(data, 1, w * h, f);
    fclose(f);
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <input_image.pgm> [output_image.pgm] [num_threads]\n", argv[0]);
        fprintf(stderr, "Example: %s input.pgm output.pgm 4\n", argv[0]);
        return 1;
    }
    
    const char *input_file = argv[1];
    const char *output_file = (argc > 2) ? argv[2] : "output.pgm";
    int threads = (argc > 3) ? atoi(argv[3]) : 4;
    
    int width, height;
    unsigned char *img = load_pgm(input_file, &width, &height);
    
    if (!img) {
        return 1;
    }
    
    printf("Loaded image: %dx%d\n", width, height);
    printf("Processing with %d threads...\n", threads);
    ordered_dither_mt(img, width, height, threads);
    save_pgm(output_file, img, width, height);
    printf("Saved to: %s\n", output_file);
    free(img);
    return 0;
}