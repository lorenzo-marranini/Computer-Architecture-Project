#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    int nDevices;
    cudaCheckError(cudaGetDeviceCount(&nDevices)); // Buona pratica controllare sempre gli errori
    
    for (int i = 0; i < nDevices; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        
        printf("=== DISPOSITIVO %d: %s ===\n", i, prop.name);
        
        // --- 1. MEMORIA GLOBALE (VRAM) ---
        // La GPU restituisce i byte totali. 
        // Dividiamo per 1024^3 per ottenere i Gigabyte.
        double vram_gb = (double)prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0);
        printf("Memoria Globale (VRAM)   : %.2f GB\n", vram_gb);
        
        // --- 2. MEMORIA CONDIVISA (SHARED MEMORY) ---
        // Dividiamo per 1024 per ottenere i Kilobyte.
        int shared_kb = prop.sharedMemPerBlock / 1024;
        printf("Memoria Condivisa/Blocco : %d KB\n", shared_kb);
        
        // (Opzionale) Memoria condivisa totale presente nel singolo SM
        int shared_sm_kb = prop.sharedMemPerMultiprocessor / 1024;
        printf("Memoria Condivisa/SM     : %d KB\n", shared_sm_kb);
        
        printf("\n--- Limiti di Calcolo ---\n");
        printf("Limite Thread per Blocco : %d\n", prop.maxThreadsPerBlock);
        printf("Numero di SM (Cores)     : %d\n", prop.multiProcessorCount);
        
        int thread_simultanei = prop.multiProcessorCount * prop.maxThreadsPerMultiProcessor;
        printf("Thread massimi in volo   : %d\n", thread_simultanei);
        printf("==================================\n\n");
    }
    return 0;
}