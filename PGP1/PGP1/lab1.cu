#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define ErrorCheck(ans) { CheckFun((ans), __FILE__, __LINE__); }

inline void CheckFun(cudaError_t code, const char *file, int line){
   if (code != cudaSuccess) {
      fprintf(stderr,"ERROR: %s %s %d\n", cudaGetErrorString(code), file, line);
      exit(0);
   }
}

__global__ void multiply(double* dev_A, double* dev_B, size_t arrLen) {
    size_t index = threadIdx.x + blockIdx.x * blockDim.x;
    while (index < arrLen) {
        dev_A[index] = dev_A[index] * dev_B[index];
        index += blockDim.x * gridDim.x;
    }
}


int main() {
    size_t arrLen;
    scanf("%zd", &arrLen);
    size_t size = sizeof(double) * arrLen;
    double *arrA = (double*)malloc(size);
    double *arrB = (double*)malloc(size);
    for (size_t i = 0; i < arrLen; ++i) {
        scanf("%lf", &arrA[i]);
    }
    for (size_t i = 0; i < arrLen; ++i) {
        scanf("%lf", &arrB[i]);
    }
    double *dev_A, *dev_B;
    ErrorCheck(cudaMalloc((void**)&dev_A, size));
    ErrorCheck(cudaMalloc((void**)&dev_B, size));
    ErrorCheck(cudaMemcpy(dev_A, arrA, size, cudaMemcpyHostToDevice));
    ErrorCheck(cudaMemcpy(dev_B, arrB, size, cudaMemcpyHostToDevice));
    dim3 blockSize = dim3(512,1,1);
    dim3 gridSize = dim3((unsigned int)arrLen / 512 + 1, 1, 1);
    multiply <<<gridSize, blockSize >>> (dev_A, dev_B, arrLen);
    ErrorCheck(cudaGetLastError());
    ErrorCheck(cudaMemcpy(arrA, dev_A, size, cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < arrLen; ++i) {
        printf("%.10lf ", arrA[i]);
    }
    printf("\n");
    free(arrA);
    free(arrB);
    ErrorCheck(cudaFree(dev_A));
    ErrorCheck(cudaFree(dev_B));

    return 0;
 }
