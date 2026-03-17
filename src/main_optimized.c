#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>
#include <ctype.h>
#include <time.h>

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
            int threshold = BAYER_MATRIX[y & 3][x & 3]; //updated
            int pixel_val = td->data[index] >> 4; //updated
            td->data[index] = (pixel_val > threshold) * 255;
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
            
            // NUOVO CODICE (Ottimizzato per ALU, Branchless e Loop Unrolled)
            int threshold = BAYER_MATRIX[y & 3][x & 3];

            // Elaboro i 3 canali esplicitamente senza il ciclo 'for (int c...)'
            int r_val = td->data[index] >> 4;
            int g_val = td->data[index + 1] >> 4;
            int b_val = td->data[index + 2] >> 4;

            // BRANCHLESS sulle tre assegnazioni
            td->data[index]     = (r_val > threshold) * 255;
            td->data[index + 1] = (g_val > threshold) * 255;
            td->data[index + 2] = (b_val > threshold) * 255;
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
    
    char magic[3];
    if (fscanf(f, "%2s", magic) != 1) {
        fclose(f);
        return NULL;
    }
    
    if (strcmp(magic, "P5") == 0) {
        *channels = 1;
    } else if (strcmp(magic, "P6") == 0) {
        *channels = 3;
    } else {
        fprintf(stderr, "Error: Invalid PGM/PPM signature\n");
        fclose(f);
        return NULL;
    }

    // --- INIZIO BLOCCO ROBUSTO PER SALTARE COMMENTI ---
    int c;
    int count = 0;
    int vals[3]; // Per contenere Larghezza, Altezza, MaxVal
    
    while (count < 3) {
        c = fgetc(f);
        if (c == EOF) break;
        
        if (isspace(c)) {
            continue; // Salta spazi, tab, a capo
        }
        
        if (c == '#') { // Se trovi un commento, salta tutta la riga
            while ((c = fgetc(f)) != '\n' && c != EOF);
            continue;
        }
        
        // Se arrivi qui, hai trovato l'inizio di un numero
        ungetc(c, f); 
        if (fscanf(f, "%d", &vals[count]) != 1) break;
        count++;
    }
    
    if (count < 3) {
        fprintf(stderr, "Error: Could not read dimensions or max value\n");
        fclose(f);
        return NULL;
    }
    
    *w = vals[0];
    *h = vals[1];
    int max_val = vals[2];
    fgetc(f); // Consuma l'ultimo spazio bianco prima dei dati binari
    // --- FINE BLOCCO ROBUSTO ---

    printf("Image dimensions: %dx%d, Channels: %d\n", *w, *h, *channels);

    // Calcolo della dimensione con cast a size_t per evitare overflow su immagini grandi
    size_t total_size = (size_t)(*w) * (*h) * (*channels);
    unsigned char *data = malloc(total_size);
    
    if (!data) {
        fprintf(stderr, "Error: Memory allocation failed for %zu bytes\n", total_size);
        fclose(f);
        return NULL;
    }
    
    size_t read = fread(data, 1, total_size, f);
    if (read != total_size) {
        fprintf(stderr, "Error: Could only read %zu of %zu bytes\n", read, total_size);
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

#include <time.h>
#include <unistd.h> // Per getcwd se vuoi debuggare i percorsi

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <input_image.pgm|.ppm> [output_image.pgm|.ppm] [num_threads]\n", argv[0]);
        fprintf(stderr, "Example: %s input.pgm output.pgm 4\n", argv[0]);
        return 1;
    }
    
    const char *input_file = argv[1];
    const char *output_file = (argc > 2) ? argv[2] : "output.pgm";
    int threads = (argc > 3) ? atoi(argv[3]) : 1;
    
    struct timespec start_total, end_total, start_compute, end_compute;
    int width, height, channels;

    // --- INIZIO TIMER TOTALE ---
    clock_gettime(CLOCK_MONOTONIC, &start_total);

    // 1. CARICAMENTO (Fase di I/O sequenziale)
    unsigned char *img = load_pgm(input_file, &width, &height, &channels);
    if (!img) {
        // Se non lo trova, stampa la cartella corrente per aiutarti a capire dove sei
        char cwd[1024];
        if (getcwd(cwd, sizeof(cwd)) != NULL) {
            fprintf(stderr, "Errore: Assicurati che '%s' sia nella cartella: %s\n", input_file, cwd);
        }
        return 1;
    }
    
    const char *format = (channels == 1) ? "grayscale" : "RGB";
    printf("Loaded image: %dx%d (%s)\n", width, height, format);
    printf("Processing with %d threads...\n", threads);

    // --- INIZIO TIMER CALCOLO PURO ---
    clock_gettime(CLOCK_MONOTONIC, &start_compute);
    
    ordered_dither_mt(img, width, height, threads, channels);
    
    clock_gettime(CLOCK_MONOTONIC, &end_compute);
    // --- FINE TIMER CALCOLO PURO ---

    // 3. SALVATAGGIO (Fase di I/O sequenziale)
    save_pgm(output_file, img, width, height, channels);

    clock_gettime(CLOCK_MONOTONIC, &end_total);
    // --- FINE TIMER TOTALE ---

    // Calcolo differenze in secondi
    double time_compute = (end_compute.tv_sec - start_compute.tv_sec) + 
                          (end_compute.tv_nsec - start_compute.tv_nsec) / 1e9;
    
    double time_total = (end_total.tv_sec - start_total.tv_sec) + 
                        (end_total.tv_nsec - start_total.tv_nsec) / 1e9;

    printf("\n================ PERFORMANCE REPORT ================\n");
    printf("Tempo di ELABORAZIONE (Dithering): %.6f secondi\n", time_compute);
    printf("Tempo TOTALE (Incluso I/O):       %.6f secondi\n", time_total);
    printf("Overhead I/O e Sistema:           %.6f secondi\n", time_total - time_compute);
    printf("Efficienza Calcolo su Totale:     %.2f%%\n", (time_compute / time_total) * 100);
    printf("====================================================\n\n");

    printf("Saved to: %s\n", output_file);
    free(img);
    return 0;
}