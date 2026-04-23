#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>
#include <stdatomic.h>
#include <stdint.h>
#include <time.h>
#include <immintrin.h>  // AVX2

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

// Histogram: plain 2D array, layout buckets[channel][value]
// (opts 1, 2, 7 removed — struct wrapper, AVX2 clear, and alignment dropped)
#define clear_histograms(b) memset((b), 0, sizeof(uint8_t) * 3 * 256)

// ---------------------------------------------------------------------------
// OPTIMIZATION 3: Vectorized median scan using AVX2.
// The original loop walked 256 buckets one at a time with branches.
// We instead use AVX2 to:
//   a) Find Z_min: first non-zero bucket (scan left to right, 32 bytes/iter)
//   b) Find Z_max: last non-zero bucket (scan right to left, 32 bytes/iter)
//   c) Find Z_med: prefix-sum to find the position >= target_pos
//      (scan left to right, horizontal add per 32-byte chunk)
// This replaces 256 scalar iterations with 8 AVX2 iterations (256/32).
// ---------------------------------------------------------------------------
static inline void find_median_stats(const uint8_t * restrict hist, int target_pos,
                                     int *out_min, int *out_max, int *out_med)
{
    __m256i zero = _mm256_setzero_si256();
    int Z_min = -1, Z_max = -1, Z_med = -1;
    int cumulative = 0;

    // Forward pass: find Z_min and Z_med together
    for (int i = 0; i < 256; i += 32) {
        __m256i chunk = _mm256_loadu_si256((const __m256i *)(hist + i));

        // Check if any byte in this chunk is non-zero
        if (!_mm256_testz_si256(chunk, chunk)) {
            // At least one non-zero — walk scalar within this chunk
            for (int j = 0; j < 32; j++) {
                uint8_t v = hist[i + j];
                if (v > 0) {
                    if (Z_min == -1) Z_min = i + j;
                    cumulative += v;
                    if (Z_med == -1 && cumulative >= target_pos)
                        Z_med = i + j;
                }
            }
        } 
    }

    int found_max = 0;
    for (int i = 256 - 32; i >= 0 && !found_max; i -= 32) {
        __m256i chunk = _mm256_loadu_si256((const __m256i *)(hist + i));
        if (!_mm256_testz_si256(chunk, chunk)) {
            for (int j = 31; j >= 0 && !found_max; j--) {
                if (hist[i + j] > 0) {
                    Z_max = i + j;
                    found_max = 1;
                }
            }
        }
    }
    *out_min = Z_min;
    *out_max = Z_max;
    *out_med = Z_med;
}

// separate the interior pixel path from the
// border pixel path at the row level, not per-pixel.
// This eliminates the branch inside the innermost loop for ~95% of pixels.
static inline void add_window_3x3_safe(const unsigned char *in, int width,
                                        int y, int x, int channels,
                                        uint8_t buckets[3][256])
{
    for (int wy = -1; wy <= 1; wy++) {
        const unsigned char *row = in + ((y + wy) * width + x - 1) * channels;
        buckets[0][row[0]]++; buckets[1][row[1]]++; buckets[2][row[2]]++;
        buckets[0][row[3]]++; buckets[1][row[4]]++; buckets[2][row[5]]++;
        buckets[0][row[6]]++; buckets[1][row[7]]++; buckets[2][row[8]]++;
    }
}

static inline void add_ring_safe(const unsigned char *in, int width,
                                  int y, int x, int r, int channels,
                                  uint8_t buckets[3][256])
{
    const unsigned char *top_row = in + ((y - r) * width + x - r) * channels;
    const unsigned char *bot_row = in + ((y + r) * width + x - r) * channels;
    int ring_width = 2 * r + 1;

    for (int wx = 0; wx < ring_width; wx++) {
        int ti = wx * channels;
        buckets[0][top_row[ti]]++;
        buckets[1][top_row[ti + 1]]++;
        buckets[2][top_row[ti + 2]]++;
        buckets[0][bot_row[ti]]++;
        buckets[1][bot_row[ti + 1]]++;
        buckets[2][bot_row[ti + 2]]++;
    }

    for (int wy = -r + 1; wy <= r - 1; wy++) {
        const unsigned char *left  = in + ((y + wy) * width + x - r) * channels;
        const unsigned char *right = in + ((y + wy) * width + x + r) * channels;
        buckets[0][left[0]]++;  buckets[1][left[1]]++;  buckets[2][left[2]]++;
        buckets[0][right[0]]++; buckets[1][right[1]]++; buckets[2][right[2]]++;
    }
}

// Clamped versions (border pixels)
static inline int clamp(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

static inline void add_window_3x3_clamped(const unsigned char *in, int width,
                                           int height, int y, int x,
                                           int channels, uint8_t buckets[3][256])
{
    for (int wy = -1; wy <= 1; wy++) {
        int ny = clamp(y + wy, 0, height - 1);
        for (int wx = -1; wx <= 1; wx++) {
            int nx = clamp(x + wx, 0, width - 1);
            int idx = (ny * width + nx) * channels;
            buckets[0][in[idx]]++;
            buckets[1][in[idx + 1]]++;
            buckets[2][in[idx + 2]]++;
        }
    }
}

static inline void add_ring_clamped(const unsigned char *in, int width,
                                     int height, int y, int x, int r,
                                     int channels, uint8_t buckets[3][256])
{
    for (int wx = -r; wx <= r; wx++) {
        int nx = clamp(x + wx, 0, width - 1);
        int ty = clamp(y - r, 0, height - 1) * width + nx;
        int by = clamp(y + r, 0, height - 1) * width + nx;
        buckets[0][in[ty * channels]]++;     buckets[1][in[ty * channels + 1]]++; buckets[2][in[ty * channels + 2]]++;
        buckets[0][in[by * channels]]++;     buckets[1][in[by * channels + 1]]++; buckets[2][in[by * channels + 2]]++;
    }
    for (int wy = -r + 1; wy <= r - 1; wy++) {
        int ny = clamp(y + wy, 0, height - 1);
        int lx = clamp(x - r, 0, width - 1);
        int rx = clamp(x + r, 0, width - 1);
        int li = (ny * width + lx) * channels;
        int ri = (ny * width + rx) * channels;
        buckets[0][in[li]]++;  buckets[1][in[li + 1]]++;  buckets[2][in[li + 2]]++;
        buckets[0][in[ri]]++;  buckets[1][in[ri + 1]]++;  buckets[2][in[ri + 2]]++;
    }
}
void* adaptive_median_worker(void* arg) {
    ThreadData* td = (ThreadData*)arg;
    int max_radius = S_MAX / 2;

    uint8_t buckets[3][256];

    while (1) {
        int start_row = atomic_fetch_add(td->global_row_counter, td->chunk_size);
        if (start_row >= td->height) break;
        int end_row = start_row + td->chunk_size;
        if (end_row > td->height) end_row = td->height;

        for (int y = start_row; y < end_row; y++) {
            // Compute safe x range once per row.
            // Interior pixels (safe_x_start..safe_x_end) never need bounds checks.
            int safe_y = (y >= max_radius && y < td->height - max_radius);

            for (int x = 0; x < td->width; x++) {
                int is_safe = safe_y &&
                              (x >= max_radius && x < td->width - max_radius);

                clear_histograms(buckets);

                int base_idx = (y * td->width + x) * td->channels;
                int Z_xy[3] = { td->in_data[base_idx],
                                 td->in_data[base_idx + 1],
                                 td->in_data[base_idx + 2] };

                int window_size = 3;
                int channels_done = 0;
                int result_color[3];

                while (window_size <= S_MAX && channels_done != 7) {
                    int r = window_size / 2;

                    if (is_safe) {
                        if (window_size == 3)
                            add_window_3x3_safe(td->in_data, td->width, y, x,
                                                td->channels, buckets);
                        else
                            add_ring_safe(td->in_data, td->width, y, x, r,
                                          td->channels, buckets);
                    } else {
                        if (window_size == 3)
                            add_window_3x3_clamped(td->in_data, td->width,
                                                   td->height, y, x,
                                                   td->channels, buckets);
                        else
                            add_ring_clamped(td->in_data, td->width, td->height,
                                             y, x, r, td->channels, buckets);
                    }

                    int target_pos = (window_size * window_size) / 2 + 1;

                    for (int c = 0; c < 3; c++) {
                        if (channels_done & (1 << c)) continue;

                        int Z_min, Z_max, Z_med;
                        find_median_stats(buckets[c], target_pos,
                                          &Z_min, &Z_max, &Z_med);

                        if ((Z_med - Z_min) > 0 && (Z_med - Z_max) < 0) {
                            result_color[c] = ((Z_xy[c] - Z_min) > 0 &&
                                               (Z_xy[c] - Z_max) < 0)
                                              ? Z_xy[c] : Z_med;
                            channels_done |= (1 << c);
                        } else if (window_size + 2 > S_MAX) {
                            result_color[c] = Z_med;
                            channels_done |= (1 << c);
                        }
                    }
                    window_size += 2;
                }

                td->out_data[base_idx]     = result_color[0];
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
    if      (strcmp(magic, "P5") == 0) *channels = 1;
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

void save_image(const char *filename, unsigned char *data, int w, int h,
                int channels) {
    FILE *f = fopen(filename, "wb");
    if (!f) return;
    fprintf(f, "%s\n%d %d\n255\n", (channels == 1) ? "P5" : "P6", w, h);
    fwrite(data, 1, w * h * channels, f);
    fclose(f);
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <input.ppm> [output.ppm] [threads]\n", argv[0]);
        return 1;
    }

    const char *input_file  = argv[1];
    const char *output_file = (argc > 2) ? argv[2] : "output.ppm";
    int num_threads         = (argc > 3) ? atoi(argv[3]) : 6;

    int width, height, channels;
    unsigned char *img_in = load_image(input_file, &width, &height, &channels);
    if (!img_in) {
        fprintf(stderr, "Errore caricamento immagine\n");
        return 1;
    }

    unsigned char *img_out = malloc(width * height * channels);

    atomic_int current_global_row = 0;
    int chunk_size = 32;

    pthread_t  threads[num_threads];
    ThreadData td[num_threads];

    printf("Filtro Mediano Adattivo AVX2 (%d thread, chunk=%d)...\n",
           num_threads, chunk_size);

    struct timespec start_time, end_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);

    for (int i = 0; i < num_threads; i++) {
        td[i] = (ThreadData){
            .in_data           = img_in,
            .out_data          = img_out,
            .width             = width,
            .height            = height,
            .channels          = channels,
            .chunk_size        = chunk_size,
            .global_row_counter = &current_global_row,
        };
        pthread_create(&threads[i], NULL, adaptive_median_worker, &td[i]);
    }

    for (int i = 0; i < num_threads; i++)
        pthread_join(threads[i], NULL);

    clock_gettime(CLOCK_MONOTONIC, &end_time);

    double elapsed = (end_time.tv_sec  - start_time.tv_sec) +
                     (end_time.tv_nsec - start_time.tv_nsec) / 1e9;
    printf("COMPUTE_TIME: %f\n", elapsed);

    free(img_in);
    free(img_out);
    return 0;
}