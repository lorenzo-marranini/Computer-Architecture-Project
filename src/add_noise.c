#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#define CLAMP(val, min, max) ((val) < (min) ? (min) : ((val) > (max) ? (max) : (val)))

#include <ctype.h>

unsigned char* load_image(const char *filename, int *w, int *h, int *channels) {
    FILE *f = fopen(filename, "rb");
    if (!f) {
        fprintf(stderr, "DEBUG [load_image]: Impossibile aprire il file '%s'. Controlla il percorso!\n", filename);
        return NULL;
    }

    char magic[3];
    if (fscanf(f, "%2s", magic) != 1) { 
        fprintf(stderr, "DEBUG [load_image]: Fallita la lettura del magic number.\n");
        fclose(f); 
        return NULL; 
    }

    if (strcmp(magic, "P5") == 0) *channels = 1;
    else if (strcmp(magic, "P6") == 0) *channels = 3;
    else { 
        fprintf(stderr, "DEBUG [load_image]: Formato non supportato '%s'. Deve essere P5 o P6.\n", magic);
        fclose(f); 
        return NULL; 
    }

    int ch;
    while ((ch = fgetc(f)) != EOF) {
        if (isspace(ch)) continue;
        if (ch == '#') {
            while ((ch = fgetc(f)) != '\n' && ch != EOF);
        } else {
            ungetc(ch, f);
            break;
        }
    }

    int max_val;
    if (fscanf(f, "%d %d %d", w, h, &max_val) != 3) { 
        fprintf(stderr, "DEBUG [load_image]: Fallita la lettura di dimensioni (W, H) o max_val.\n");
        fclose(f); 
        return NULL; 
    }
    
    fgetc(f); 

    size_t size = (size_t)(*w) * (*h) * (*channels);
    unsigned char *data = malloc(size);
    if (!data) { 
        fprintf(stderr, "DEBUG [load_image]: Errore malloc per %zu bytes.\n", size);
        fclose(f); 
        return NULL; 
    }

    if (fread(data, 1, size, f) != size) {
        fprintf(stderr, "DEBUG [load_image]: Attenzione, i dati immagine letti sono inferiori al previsto.\n");
    }

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

typedef struct {
    int x;
    int y;
    int type; 
    double radius;
} NoiseSplat;

int main(int argc, char *argv[]) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <clean_input.ppm> <noisy_output.ppm>\n", argv[0]);
        return 1;
    }
    
    srand(time(NULL));
    
    int width, height, channels;
    unsigned char *img = load_image(argv[1], &width, &height, &channels);
    if (!img) {
        fprintf(stderr, "Errore caricamento immagine\n");
        return 1;
    }

    long long total_pixels = width * height;
    long long corrupted_pixels = 0;
    
    // Aumentato leggermente per compensare la mancanza di strisce in basso
    int base_num_splats = (width * height) / 60; 
    NoiseSplat* splats = malloc(base_num_splats * 2 * sizeof(NoiseSplat));
    int actual_num_splats = 0;
    
    // --- DISTRIBUZIONE SBILANCIATA (Ampiezza Decrescente) ---
    for (int i = 0; i < base_num_splats; i++) {
        int candidate_y = rand() % height;
        double normalized_y = (double)candidate_y / height;
        
        // IL SEGRETO: L'ampiezza decresce da 1.0 (cima) a 0.0 (fondo).
        // L'onda sinusoidale perde forza man mano che scende.
        double amplitude = 1.0 - normalized_y; 
        double sine_val = sin(normalized_y * 7.0 * 3.14159265) * amplitude;
        
        // Sotto y=0.6, l'onda è così debole (amplitude < 0.4) che sine_val
        // non supererà MAI la soglia di 0.4 o -0.4. Il fondo rimane pulito!
        double spawn_chance = 0.05; // Rumore base ridotto per evidenziare il contrasto
        if (sine_val > 0.4 || sine_val < -0.4) {
            spawn_chance = 0.95; // Pioggia di splats nelle zone dense superiori!
        }
        
        double r = (double)rand() / RAND_MAX;
        if (r < spawn_chance) {
            splats[actual_num_splats].x = rand() % width;
            splats[actual_num_splats].y = candidate_y;
            splats[actual_num_splats].type = (rand() % 2 == 0) ? 0 : 255;
            splats[actual_num_splats].radius = 2.0 + (rand() % 5); 
            actual_num_splats++;
        }
    }

    float *prob_map = malloc(width * height * sizeof(float));
    int *type_map = malloc(width * height * sizeof(int));
    
    for(int i = 0; i < width * height; i++) {
        prob_map[i] = 0.50f; // Fondo pulito (10%)
        type_map[i] = -1; 
    }
    
    for (int i = 0; i < actual_num_splats; i++) {
        int r_int = (int)splats[i].radius + 1;
        int min_x = CLAMP(splats[i].x - r_int, 0, width - 1);
        int max_x = CLAMP(splats[i].x + r_int, 0, width - 1);
        int min_y = CLAMP(splats[i].y - r_int, 0, height - 1);
        int max_y = CLAMP(splats[i].y + r_int, 0, height - 1);
        
        double r_sq = splats[i].radius * splats[i].radius;
        
        for (int y = min_y; y <= max_y; y++) {
            for (int x = min_x; x <= max_x; x++) {
                double dx = x - splats[i].x;
                double dy = y - splats[i].y;
                double dist_sq = (dx * dx) + (dy * dy);
                
                if (dist_sq < r_sq) {
                    double intensity = 1.0 - (dist_sq / r_sq);
                    double splat_prob = 0.95 * intensity;
                    
                    int idx = y * width + x;
                    if (splat_prob > prob_map[idx]) {
                        prob_map[idx] = splat_prob;
                        type_map[idx] = splats[i].type;
                    }
                }
            }
        }
    }

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = y * width + x;
            double r = (double)rand() / RAND_MAX;
            
            if (r < prob_map[idx]) {
                int noise_val;
                if (type_map[idx] != -1 && prob_map[idx] > 0.50) {
                    noise_val = type_map[idx]; 
                } else {
                    noise_val = (rand() % 2 == 0) ? 0 : 255; 
                }
                
                for (int c = 0; c < channels; c++) {
                    img[idx * channels + c] = noise_val;
                }
                corrupted_pixels++;
            }
        }
    }

    save_image(argv[2], img, width, height, channels);
    
    double actual_coverage = ((double)corrupted_pixels / total_pixels) * 100.0;
    
    printf("✅ Unbalanced Banded Micro-Splat Complete!\n");
    printf("📊 Total Noise Coverage: %.2f%%\n", actual_coverage);
    printf("Output saved to: %s\n", argv[2]);

    free(splats);
    free(prob_map);
    free(type_map);
    free(img);
    return 0;
}