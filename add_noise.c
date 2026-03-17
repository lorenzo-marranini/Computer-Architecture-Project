#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <ctype.h>

// This is the fix! It skips whitespace and any hidden "#" comments in the image header
void skip_comments(FILE *f) {
    int ch;
    while ((ch = fgetc(f)) != EOF) {
        if (isspace(ch)) {
            continue;
        }
        if (ch == '#') {
            while ((ch = fgetc(f)) != '\n' && ch != EOF); // Skip to end of line
        } else {
            ungetc(ch, f); // Put the character back, it's actual data
            break;
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("Usage: %s <clean_input.ppm> <noisy_output.ppm>\n", argv[0]);
        return 1;
    }

    printf("Opening %s...\n", argv[1]);
    FILE *f_in = fopen(argv[1], "rb");
    if (!f_in) {
        printf("Error: Could not open input file!\n");
        return 1;
    }

    char magic[3];
    fscanf(f_in, "%2s", magic);
    skip_comments(f_in);

    int w, h, max_val;
    fscanf(f_in, "%d %d", &w, &h);
    skip_comments(f_in);
    fscanf(f_in, "%d", &max_val);
    fgetc(f_in); // Consume the exact single byte of whitespace after max_val

    printf("Header parsed successfully: %dx%d, Magic: %s\n", w, h, magic);

    int channels = (strcmp(magic, "P6") == 0) ? 3 : 1;
    unsigned char *img = malloc(w * h * channels);
    if (!img) {
        printf("Error: Memory allocation failed!\n");
        fclose(f_in);
        return 1;
    }

    size_t read_bytes = fread(img, 1, w * h * channels, f_in);
    printf("Read %zu bytes from image.\n", read_bytes);
    fclose(f_in);

    printf("Adding heavy salt and pepper noise to the bottom half...\n");
    srand(time(NULL));
    int half_height = h / 2;
    
    for (int y = half_height; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int idx = (y * w + x) * channels;
            
            int random_val = rand() % 100; 
            if (random_val < 10) { // 10% Pepper
                for(int c=0; c<channels; c++) img[idx+c] = 0;
            } 
            else if (random_val > 90) { // 10% Salt
                for(int c=0; c<channels; c++) img[idx+c] = 255;
            }
        }
    }

    printf("Saving to %s...\n", argv[2]);
    FILE *f_out = fopen(argv[2], "wb");
    if (!f_out) {
        printf("Error: Could not open output file for writing!\n");
        free(img);
        return 1;
    }
    
    fprintf(f_out, "%s\n%d %d\n255\n", magic, w, h);
    fwrite(img, 1, w * h * channels, f_out);
    fclose(f_out);
    
    free(img);
    printf("Done! Successfully created %s\n", argv[2]);
    return 0;
}