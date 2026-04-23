#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>
#include <time.h>      // Required for clock_gettime
#include <stdatomic.h> // Required for atomic_int and atomic_fetch_add

// Maximum window size for the adaptive filter (must be an odd number).
// A larger S_MAX means a heavier potential workload for noisy pixels.
const int S_MAX = 15; 

typedef struct {
    unsigned char *in_data;
    unsigned char *out_data;
    int width;
    int height;
    int channels;
    int start_row;
    int end_row;
    
    // Pointers and data for our shared work pool
    atomic_int *tasks_completed; 
    int total_tasks;             
} ThreadData;


// The Academic Adaptive Median Filter Worker
void* adaptive_median_worker(void* arg) {
    ThreadData *td = (ThreadData *)arg;
    
    while (1) {
        int current_task = atomic_fetch_add(td->tasks_completed, 1);
        
        if (current_task >= td->total_tasks) {
            break; 
        }

        volatile int a = 0; 
        
        for(long long i = 0; i < 200000000LL; i++){
            a = a + i * a - 10;
            a *= a;
            a %=100000000;        
            __asm__ volatile("" ::: "memory");
        }
        
        printf("Task %d completed. finito: %f\n", current_task, a);
    }
    
    return NULL;
}

int main(int argc, char *argv[]) {
    // Require two arguments: num_threads and total_tasks (x)
    if (argc < 3) {
        printf("Usage: %s <num_threads> <total_tasks>\n", argv[0]);
        return 1;
    }

    int num_threads = atoi(argv[1]); 
    int total_tasks = atoi(argv[2]); 
    
    printf("Threads: %d, Total Tasks: %d\n", num_threads, total_tasks);
    
    pthread_t threads[num_threads];
    ThreadData td[num_threads];
    
    // Initialize the shared atomic counter
    atomic_int shared_counter = 0;

    struct timespec start_time, end_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);
    
    for (int i = 0; i < num_threads; i++) {
        // Point the thread data to our shared atomic counter
        td[i].tasks_completed = &shared_counter;
        td[i].total_tasks = total_tasks;
        
        pthread_create(&threads[i], NULL, adaptive_median_worker, &td[i]);
    }

    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }
    
    clock_gettime(CLOCK_MONOTONIC, &end_time);

    double elapsed = (end_time.tv_sec - start_time.tv_sec) + 
                     (end_time.tv_nsec - start_time.tv_nsec) / 1e9;

    // We print a specific tag "COMPUTE_TIME:" so Python can find it easily
    printf("COMPUTE_TIME: %f\n", elapsed);
    return 0;
}