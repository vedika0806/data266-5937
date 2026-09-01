
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 16

__global__ void matMulKernel(float *A, float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

void matMulCPU(float *A, float *B, float *C, int N) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++)
                sum += A[i * N + k] * B[k * N + j];
            C[i * N + j] = sum;
        }
}

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("Usage: %s <matrix_size>\n", argv[0]);
        return 1;
    }
    int N = atoi(argv[1]);
    size_t bytes = N * N * sizeof(float);

    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C_gpu = (float*)malloc(bytes);
    float *h_C_cpu = (float*)malloc(bytes);

    srand(5937);
    for (int i = 0; i < N * N; i++) {
        h_A[i] = (float)(rand() % 100) / 10.0f;
        h_B[i] = (float)(rand() % 100) / 10.0f;
    }

    clock_t cpu_start = clock();
    matMulCPU(h_A, h_B, h_C_cpu, N);
    clock_t cpu_end = clock();
    double cpu_time_ms = 1000.0 * (cpu_end - cpu_start) / CLOCKS_PER_SEC;

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaEvent_t startH2D, stopH2D, startKernel, stopKernel, startD2H, stopD2H;
    cudaEventCreate(&startH2D); cudaEventCreate(&stopH2D);
    cudaEventCreate(&startKernel); cudaEventCreate(&stopKernel);
    cudaEventCreate(&startD2H); cudaEventCreate(&stopD2H);

    cudaEventRecord(startH2D);
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
    cudaEventRecord(stopH2D);
    cudaEventSynchronize(stopH2D);
    float h2d_ms = 0;
    cudaEventElapsedTime(&h2d_ms, startH2D, stopH2D);

    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
    dim3 gridDim((N + TILE_WIDTH - 1) / TILE_WIDTH, (N + TILE_WIDTH - 1) / TILE_WIDTH);

    // ---- WARM-UP LAUNCH (NOT TIMED) ----
    // Absorbs CUDA context creation / kernel JIT compilation cost so it
    // doesn't contaminate the timed measurement below.
    matMulKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    // ---- TIMED KERNEL LAUNCH ----
    cudaEventRecord(startKernel);
    matMulKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stopKernel);
    cudaEventSynchronize(stopKernel);
    float kernel_ms = 0;
    cudaEventElapsedTime(&kernel_ms, startKernel, stopKernel);

    cudaEventRecord(startD2H);
    cudaMemcpy(h_C_gpu, d_C, bytes, cudaMemcpyDeviceToHost);
    cudaEventRecord(stopD2H);
    cudaEventSynchronize(stopD2H);
    float d2h_ms = 0;
    cudaEventElapsedTime(&d2h_ms, startD2H, stopD2H);

    float transfer_ms = h2d_ms + d2h_ms;
    float gpu_end_to_end_ms = h2d_ms + kernel_ms + d2h_ms;
    float speedup = (float)cpu_time_ms / gpu_end_to_end_ms;

    float maxDiff = 0.0f;
    for (int i = 0; i < N * N; i++) {
        float diff = fabsf(h_C_cpu[i] - h_C_gpu[i]);
        if (diff > maxDiff) maxDiff = diff;
    }

    printf("N=%d | CPU: %.3f ms | GPU kernel: %.3f ms | H2D+D2H: %.3f ms | End-to-end GPU: %.3f ms | Speedup: %.3fx | MaxDiff: %f\n",
           N, cpu_time_ms, kernel_ms, transfer_ms, gpu_end_to_end_ms, speedup, maxDiff);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C_gpu); free(h_C_cpu);

    return 0;
}
