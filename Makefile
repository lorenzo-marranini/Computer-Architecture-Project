# Compiler and flags
CC = gcc
CFLAGS = -O3 -march=native -Wall -Wextra -g -pthread -I/opt/amduprof/include
LDFLAGS = -lm -L/opt/amduprof/lib/x64/shared -Wl,-rpath=/opt/amduprof/lib/x64/shared -lAMDProfileController

# Directories
SRC_DIR = src

# Executable names
TARGETS = add_noise blur denoise dither denoise_calendar_3_lock denoise_calendar_3_lock_debug add_noise_2_rows  dither_opt

# Default target runs when you just type 'make'
.PHONY: all clean
all: $(TARGETS)

# Compile add_noise
add_noise: $(SRC_DIR)/add_noise.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)
add_noise_2_rows: $(SRC_DIR)/add_noise_2_rows.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

# Compile blur (from gauss.c)
blur: $(SRC_DIR)/gauss.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)


# Compile denoise
denoise_calendar_3_lock: $(SRC_DIR)/denoise_calendar_3_lock.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

denoise_calendar_3_lock_debug: $(SRC_DIR)/denoise_calendar_3_lock_debug.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

denoise: $(SRC_DIR)/denoise.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

# Compile dither (from main.c)
dither: $(SRC_DIR)/main.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

# Compile optimized dither (from main_optimized.c)
dither_opt: $(SRC_DIR)/main_optimized.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

# Clean up compiled binaries
clean:
	rm -f $(TARGETS)