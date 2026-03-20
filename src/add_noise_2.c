#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

// ==========================================
// IMPOSTA QUI IL LIVELLO DI RUMORE (da 1 a 10)
// 1 = 5% di pixel corrotti
// 10 = 50% di pixel corrotti
const int NOISE_LEVEL = 5; 
// ==========================================

// Funzione per caricare un'immagine PPM (ignorando eventuali commenti)
unsigned char* load_ppm(const char *filename, int *w, int *h, int *channels) {
    FILE *f = fopen(filename, "rb");
    if (!f) return NULL;
    
    char magic[3];
    if (fscanf(f, "%2s", magic) != 1) { fclose(f); return NULL; }
    
    if (strcmp(magic, "P6") == 0) {
        *channels = 3;
    } else {
        printf("Supportato solo il formato P6 (a colori).\n");
        fclose(f); 
        return NULL;
    }

    // Salta eventuali commenti nel file PPM
    int c = getc(f);
    while (c == '#' || c == ' ' || c == '\n' || c == '\r') {
        if (c == '#') {
            while (getc(f) != '\n'); // Salta fino alla fine della riga
        }
        c = getc(f);
    }
    ungetc(c, f);

    int max_val;
    if (fscanf(f, "%d %d %d", w, h, &max_val) != 3) { fclose(f); return NULL; }
    fgetc(f); // Consuma il carattere newline
    
    unsigned char *data = malloc(*w * *h * *channels);
    fread(data, 1, *w * *h * *channels, f);
    fclose(f);
    return data;
}

// Funzione per salvare l'immagine
void save_ppm(const char *filename, unsigned char *data, int w, int h, int channels) {
    FILE *f = fopen(filename, "wb");
    if (!f) {
        printf("Errore nel salvataggio del file.\n");
        return;
    }
    fprintf(f, "P6\n%d %d\n255\n", w, h);
    fwrite(data, 1, w * h * channels, f);
    fclose(f);
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("Uso: %s <input_pulito.ppm> <output_rumoroso.ppm>\n", argv[0]);
        return 1;
    }

    const char *input_file = argv[1];
    const char *output_file = argv[2];
    
    int width, height, channels;
    unsigned char *img = load_ppm(input_file, &width, &height, &channels);
    
    if (!img) {
        printf("Errore: impossibile caricare %s. Assicurati che sia un PPM (P6).\n", input_file);
        return 1;
    }

    // Usiamo la costante invece dell'input da tastiera
    int livello = NOISE_LEVEL;

    // Controllo di sicurezza: limitiamo il valore tra 1 e 10
    if (livello < 1) livello = 1;
    if (livello > 10) livello = 10;

    int probabilita = livello * 5; 

    // Inizializza il generatore di numeri casuali
    srand(time(NULL));

    int totale_pixel = width * height;
    int pixel_corrotti = 0;

    // Aggiunta del rumore
    for (int i = 0; i < totale_pixel; i++) {
        int rand_val = rand() % 100;

        if (rand_val < probabilita) {
            int idx = i * channels;
            
            // Rumore a colori casuali
            img[idx]     = rand() % 256; // Red
            img[idx + 1] = rand() % 256; // Green
            img[idx + 2] = rand() % 256; // Blue
            
            pixel_corrotti++;
        }
    }

    printf("Livello rumore impostato a: %d\n", livello);
    printf("Elaborazione completata. %d pixel alterati su %d (circa %d%%).\n", 
           pixel_corrotti, totale_pixel, probabilita);

    save_ppm(output_file, img, width, height, channels);
    printf("Immagine rumorosa salvata in: %s\n", output_file);
    
    free(img);
    return 0;
}