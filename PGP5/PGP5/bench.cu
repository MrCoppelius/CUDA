
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <math.h>


#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

#define ErrorCheck(ans) { CheckFun((ans), __FILE__, __LINE__); }

inline void CheckFun(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "ERROR: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(0);
    }
}

inline uint32_t nextPowTwo(uint32_t n) {
    --n;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    return ++n;
}
__global__ void bitonicSort(int * devArr, const uint32_t mergeStep, const uint32_t step, const uint32_t size) {
    uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x,
        offsetx = blockDim.x * gridDim.x;
    for (uint32_t n = idx; n < size; n += offsetx) {
        uint32_t nPlusStep = n ^ step;
        if (nPlusStep > n) {
            if (((n&mergeStep) == 0) && (devArr[n] > devArr[nPlusStep])) {
                int32_t tmp = devArr[n];
                devArr[n] = devArr[nPlusStep];
                devArr[nPlusStep] = tmp;

            }
            else if (((n&mergeStep) != 0) && (devArr[n] < devArr[nPlusStep])) {
                int32_t tmp = devArr[n];
                devArr[n] = devArr[nPlusStep];
                devArr[nPlusStep] = tmp;
            }
        }
    }
    return;
}

__global__ void bitonicSortShared(int * devArr, const uint32_t mergeStep, const uint32_t step, const uint32_t size) {
    uint32_t idx = threadIdx.x;
    __shared__  int32_t devArrShared[512 * 8];
    for (uint32_t i = idx; i < size; i += 512) {
        devArrShared[i] = devArr[i];
    }
    __syncthreads();
    for (uint32_t n = idx; n < size; n += 512) {
        uint32_t nPlusStep = n ^ step;

        if (nPlusStep > n) {
            if (((n&mergeStep) == 0) && (devArrShared[n] > devArrShared[nPlusStep])) {
                int32_t tmp = devArrShared[n];
                devArrShared[n] = devArrShared[nPlusStep];
                devArrShared[nPlusStep] = tmp;

            }
            else if (((n&mergeStep) != 0) && (devArrShared[n] < devArrShared[nPlusStep])) {
                int32_t tmp = devArrShared[n];
                devArrShared[n] = devArrShared[nPlusStep];
                devArrShared[nPlusStep] = tmp;
            }
        }
    }
    __syncthreads();
    for (uint32_t i = idx; i < size; i += 512) {
        devArr[i] = devArrShared[i];
    }
    return;
}



__host__ void bitonicSort(int32_t * arr, uint32_t roundSize, uint32_t size) {

    int32_t * devArr;
    ErrorCheck(cudaMalloc(&devArr, roundSize * sizeof(int32_t)));
    ErrorCheck(cudaMemcpy(devArr, arr, roundSize * sizeof(int32_t), cudaMemcpyHostToDevice));
    bool flag = false; //roundSize <= 512 * 8;
    cudaEvent_t start, stop;
    float gpuTime = 0.0f;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    for (uint32_t mergeStep = 2; mergeStep <= roundSize; mergeStep <<= 1) {
        for (uint32_t step = mergeStep >> 1; step > 0; step >>= 1) {

            if (flag)
                bitonicSortShared <<<1, 512 >>> (devArr, mergeStep, step, roundSize);
            else
                bitonicSort <<<1, 32 >>> (devArr, mergeStep, step, roundSize);
        }
        ErrorCheck(cudaGetLastError());
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpuTime, start, stop);
    printf("time spent executing by the GPU: %.2f millseconds\n", gpuTime);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    ErrorCheck(cudaMemcpy(arr, devArr, size * sizeof(int32_t), cudaMemcpyDeviceToHost));
    ErrorCheck(cudaFree(devArr));
    return;
}



int main() {
#ifdef _WIN32
    _setmode(_fileno(stdin), O_BINARY);
    _setmode(_fileno(stdout), O_BINARY);
#endif
    uint32_t size, roundSize;
    int32_t * arr;
    fread(&size, sizeof(uint32_t), 1, stdin);
    roundSize = nextPowTwo(size);
    arr = (int32_t*)malloc(sizeof(int32_t) * roundSize);

    fread(arr, sizeof(uint32_t), size, stdin);
    for (uint32_t i = size; i < roundSize; ++i) {
        arr[i] = INT32_MAX;
    }
    bitonicSort(arr, roundSize, size);
    //fwrite(arr, sizeof(int32_t), size, stdout);
    return 0;
}

