#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

// ==========================================
// IMPOSTA QUI LE STRISCE IN CUI APPLICARE IL RUMORE
// L'immagine viene divisa in 10 fasce orizzontali (dall'alto verso il basso).
// Metti 1 per applicare il rumore in quella fascia, 0 per lasciarla pulita.
const int STRISCE_ATTIVE[10] = {
    1, // Striscia 0 (In alto)
    0, // Striscia 1
    1, // Striscia 2
    0, // Striscia 3
    1, // Striscia 4
    0, // Striscia 5
    1, // Striscia 6
    0, // Striscia 7
    0, // Striscia 8
    1  // Striscia 9 (In basso)
};

// IMPOSTA QUI IL LIVELLO DI RUMORE (da 1 a 10) 
// (Applica questa probabilità SOLO alle strisce impostate a 1)
const int NOISE_LEVEL = 8; 
// ==========================================

// Funzione per caricare un'immagine PPM
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
            while (getc(f) != '\n'); 
        }
        c = getc(f);
    }
    ungetc(c, f);

    int max_val;
    if (fscanf(f, "%d %d %d", w, h, &max_val) != 3) { fclose(f); return NULL; }
    fgetc(f); 
    
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
        printf("Uso: %s <input_pulito.ppm> <output_strisce_rumorose.ppm>\n", argv[0]);
        return 1;
    }

    const char *input_file = argv[1];
    const char *output_file = argv[2];
    
    int width, height, channels;
    unsigned char *img = load_ppm(input_file, &width, &height, &channels);
    
    if (!img) {
        printf("Errore: impossibile caricare %s.\n", input_file);
        return 1;
    }

    int livello = NOISE_LEVEL;
    if (livello < 1) livello = 1;
    if (livello > 10) livello = 10;
    
    // Probabilità percentuale di alterare un pixel
    int probabilita = livello * 5; 

    // Inizializza il generatore random
    srand(time(NULL));

    int totale_pixel = width * height;
    int pixel_corrotti = 0;

    // Applichiamo il rumore riga per riga
    for (int y = 0; y < height; y++) {
        // Calcoliamo in quale delle 10 strisce ci troviamo.
        int indice_striscia = (y * 10) / height;

        // Se la striscia corrente è "accesa" nell'array, applichiamo il rumore
        if (STRISCE_ATTIVE[indice_striscia] == 1) {
            
            for (int x = 0; x < width; x++) {
                int rand_val = rand() % 100;

                // Se il pixel cade nella probabilità di essere corrotto...
                if (rand_val < probabilita) {
                    int idx = (y * width + x) * channels;
                    
                    // Rumore Colorato Estremo (Salt & Pepper su ogni canale)
                    // Ogni canale diventa indipendentemente 0 oppure 255
                    img[idx]     = (rand() % 2 == 1) ? 255 : 0; // Canale Rosso
                    img[idx + 1] = (rand() % 2 == 1) ? 255 : 0; // Canale Verde
                    img[idx + 2] = (rand() % 2 == 1) ? 255 : 0; // Canale Blu
                    
                    pixel_corrotti++;
                }
            }
        }
    }

    printf("Livello rumore nelle strisce attive: %d\n", livello);
    printf("Elaborazione completata. %d pixel alterati con rumore colorato estremo.\n", pixel_corrotti);

    save_ppm(output_file, img, width, height, channels);
    printf("Immagine a strisce rumorose salvata in: %s\n", output_file);
    
    free(img);
    return 0;
}