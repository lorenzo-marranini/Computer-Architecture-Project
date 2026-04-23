# Compiler and flags
CC = gcc
CFLAGS = -O3 -march=native -mtune=native -Wall -Wextra -g -pthread  -flto -funroll-loops #-I/opt/amduprof/include
LDFLAGS = -lm #-L/opt/amduprof/lib/x64/shared -Wl,-rpath=/opt/amduprof/lib/x64/shared -lAMDProfileController

# Directories
SRC_DIR = src

# Executable names
TARGETS = add_noise denoise_calendar_final denoise denoise_calendar_3_lock denoise_calendar_3_lock_debug denoise_calendar_3 add_noise_2_rows  denoise_calendar_final_float denoise_float_useless denoise_vectorized # denoise_2_gpu denoise_1_gpu

# Default target runs when you just type 'make'
.PHONY: all clean
all: $(TARGETS)

# Compile add_noise
denoise_vectorized: $(SRC_DIR)/denoise_vectorized.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

denoise_1_gpu: $(SRC_DIR)/denoise_1_gpu.cu
	nvcc -O3 -arch=sm_75 $< -o $@ $(LDFLAGS)


denoise_2_gpu: $(SRC_DIR)/denoise_2_gpu.cu
	nvcc -O3 -arch=sm_75 $< -o $@ $(LDFLAGS)


denoise_float_useless: $(SRC_DIR)/denoise_float_useless.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

add_noise: $(SRC_DIR)/add_noise.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)
add_noise_2_rows: $(SRC_DIR)/add_noise_2_rows.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

denoise_calendar_final_float: $(SRC_DIR)/denoise_calendar_final_float.c
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