# Compiler and flags
CC = gcc
CFLAGS = -O3 -march=native -mtune=native -Wall -Wextra -g -pthread -I/opt/amduprof/include -flto -funroll-loops
LDFLAGS = -lm -L/opt/amduprof/lib/x64/shared -Wl,-rpath=/opt/amduprof/lib/x64/shared -lAMDProfileController

# Directories
SRC_DIR = src

# Executable names
TARGETS = add_noise denoise_calendar_final denoise denoise_calendar_3_lock denoise_calendar_3_lock_debug denoise_calendar_3 add_noise_2_rows  

# Default target runs when you just type 'make'
.PHONY: all clean
all: $(TARGETS)

# Compile add_noise
add_noise: $(SRC_DIR)/add_noise.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)
add_noise_2_rows: $(SRC_DIR)/add_noise_2_rows.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)


denoise_calendar_final: $(SRC_DIR)/denoise_calendar_final.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)
# Compile denoise
denoise_calendar_3_lock: $(SRC_DIR)/denoise_calendar_lock.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)
denoise_calendar_3: $(SRC_DIR)/denoise_calendar.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

denoise_calendar_3_lock_debug: $(SRC_DIR)/denoise_calendar_3_lock_debug.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

denoise: $(SRC_DIR)/denoise.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)


# Clean up compiled binaries
clean:
	rm -f $(TARGETS)