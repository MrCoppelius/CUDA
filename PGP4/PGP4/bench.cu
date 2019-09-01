/*#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>
#include <math.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <thrust/device_vector.h>
#include <thrust/extrema.h>




#define ErrorCheck(ans) { CheckFun((ans), __FILE__, __LINE__); }

inline void CheckFun(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "ERROR: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(0);
    }
}

struct cmp {
    __host__ __device__
        bool operator()(double lhs, double rhs) const {
        return fabs(lhs) < fabs(rhs);
    }
};

__global__ void rowsPermutation(double * __restrict__  matrix, const uint32_t matrixDim, const uint64_t pitch,
    const uint64_t midInColumnID, const uint64_t maxInColumnID) {
    uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x + midInColumnID;
    uint32_t offsetx = blockDim.x * gridDim.x;
    for (uint32_t i = idx; i < matrixDim; i += offsetx) {
        double tmp = *((double*)((char*)matrix + pitch * i) + midInColumnID);
        *((double*)((char*)matrix + pitch * i) + midInColumnID) = *((double*)((char*)matrix + pitch * i) + maxInColumnID);
        *((double*)((char*)matrix + pitch * i) + maxInColumnID) = tmp;
    }
    return;
}

__global__ void updateBotRows(double * __restrict__  matrix, const uint32_t matrixDim, const uint64_t pitch,
    const uint64_t midInColumnID, const double midInColumnVal) {
    uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x + midInColumnID + 1;
    uint32_t idy = threadIdx.y + blockIdx.y * blockDim.y + midInColumnID + 1;
    uint32_t offsetx = blockDim.x * gridDim.x;
    uint32_t offsety = blockDim.y * gridDim.y;
    double factor;
    for (uint32_t j = idy; j < matrixDim; j += offsety) {
        factor = *((double*)((char*)matrix + pitch * midInColumnID) + j);
        if (fabs(factor) < 1e-7) continue;
        for (uint32_t i = idx; i < matrixDim; i += offsetx) {
            *((double*)((char*)matrix + pitch * i) + j) -= *((double*)((char*)matrix + pitch * i) + midInColumnID) * factor / midInColumnVal;
        }
    }
    return;
}

__host__ double findDet(double * __restrict__  matrix, const uint32_t matrixDim) {
    double det = 1;
    double *matrixDev;
    uint64_t devPitch, hostPitch;
    hostPitch = sizeof(double) * matrixDim;
    cudaMallocPitch(&matrixDev, &devPitch, matrixDim * sizeof(double), matrixDim);
    cudaMemcpy2D(matrixDev, devPitch, matrix, hostPitch, sizeof(double) * matrixDim, matrixDim, cudaMemcpyHostToDevice);
    cudaEvent_t start, stop;
    float gpuTime = 0.0f;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    for (uint32_t i = 0; i < matrixDim; ++i) {
        thrust::device_ptr<double> currColumnPtr((double*)((char*)matrixDev + devPitch * i));
        thrust::device_ptr<double> start((double*)((char*)matrixDev + devPitch * i) + i);
        thrust::device_ptr<double> end((double*)((char*)matrixDev + devPitch * i) + matrixDim);
        thrust::device_ptr<double> maxInColumnPtr = thrust::max_element(start, end, cmp());
        uint64_t maxInColumnID = (uint64_t)(maxInColumnPtr - currColumnPtr);
        double maxInColumnVal = *maxInColumnPtr;
        det *= maxInColumnVal;
        if (fabs(maxInColumnVal) < 1e-7) {
            det = 0;
            break;
        }
        if (maxInColumnID != i) {
            det *= -1;
            rowsPermutation << <dim3(1), dim3(32) >> > (matrixDev, matrixDim, devPitch, i, maxInColumnID);
        }

        if (i != matrixDim - 1) {
            updateBotRows << <dim3(1, 1), dim3(16, 32) >> > (matrixDev, matrixDim, devPitch, i, maxInColumnVal);
        }

    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpuTime, start, stop);
    printf("time spent executing by the GPU: %.2f millseconds\n", gpuTime);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(matrixDev);
    return det;
}


int main() {

    uint32_t matrixDim;
    double *matrix;
    scanf("%" SCNu32, &matrixDim);
    matrix = (double*)malloc(sizeof(double) * matrixDim * matrixDim);
    for (uint32_t i = 0; i < matrixDim; ++i) {
        for (uint32_t j = 0; j < matrixDim; ++j) {
            scanf("%lf", &matrix[j * matrixDim + i]);
        }
    }
    double det = findDet(matrix, matrixDim);
    if (matrixDim == 0) det = 0;
    printf("%.10e\n", det);
    free(matrix);
    return 0;
}*/