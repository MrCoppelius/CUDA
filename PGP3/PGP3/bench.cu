#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>

#define ErrorCheck(ans) { CheckFun((ans), __FILE__, __LINE__); }

inline void CheckFun(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "ERROR: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(0);
    }
}

__constant__ double3 centerClusters[32];

__device__ inline double calculateDistance(uchar4 &A, double3 &B) {
    return  sqrt((double)(A.x - B.x)*(A.x - B.x) + (A.y - B.y)*(A.y - B.y) + (A.z - B.z)*(A.z - B.z));
}

__global__ void KMeans(uchar4 * __restrict__ img, const uint32_t w, const uint32_t h, const uint32_t nc) {
    uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t idy = threadIdx.y + blockIdx.y * blockDim.y;
    uint32_t offsetx = blockDim.x * gridDim.x;
    uint32_t offsety = blockDim.y * gridDim.y;
    for (uint32_t i = idx; i < w; i += offsetx) {
        for (uint32_t j = idy; j < h; j += offsety) {
            double distanceMin = calculateDistance(img[j * w + i], centerClusters[0]);
            uint32_t clusterNumber = 0;
            for (uint32_t k = 1; k < nc; ++k) {
                double distanceTmp = calculateDistance(img[j * w + i], centerClusters[k]);
                if (distanceTmp < distanceMin) {
                    distanceMin = distanceTmp;
                    clusterNumber = k;
                }
            }
            img[j * w + i].w = clusterNumber;
        }
    }
}


__host__ bool updateClusters(uchar4 * __restrict__ img, uchar4 * __restrict__ imgNew,
    double3 * __restrict__ centerClustersHost, const uint32_t w, const uint32_t h, const uint32_t nc) {
    uint64_t countElementOnCluster[32] = { 0 };
    ulonglong3 sumElementOnCluster[32] = { make_ulonglong3(0, 0, 0) };
    bool notEqual = false;
    for (uint32_t i = 0; i < w*h; ++i) {
        if (imgNew[i].w != img[i].w) notEqual = true;
        countElementOnCluster[imgNew[i].w]++;
        sumElementOnCluster[imgNew[i].w].x += imgNew[i].x;
        sumElementOnCluster[imgNew[i].w].y += imgNew[i].y;
        sumElementOnCluster[imgNew[i].w].z += imgNew[i].z;

    }
    for (uint32_t i = 0; i < nc; ++i) {
        centerClustersHost[i].x = (double)sumElementOnCluster[i].x / (double)countElementOnCluster[i];
        centerClustersHost[i].y = (double)sumElementOnCluster[i].y / (double)countElementOnCluster[i];
        centerClustersHost[i].z = (double)sumElementOnCluster[i].z / (double)countElementOnCluster[i];
    }
    return notEqual;
}

__host__ void KMeans(uchar4 * __restrict__ img, uchar4 * __restrict__ imgNew,
    double3 * __restrict__ centerClustersHost, const uint32_t w, const uint32_t h, const uint32_t nc) {
    uchar4 *imgDev;
    ErrorCheck(cudaMalloc(&imgDev, sizeof(uchar4) * w * h));
    ErrorCheck(cudaMemcpy(imgDev, img, sizeof(uchar4) * w * h, cudaMemcpyHostToDevice));
    bool flag = true;
    while (flag) {
        ErrorCheck(cudaMemcpyToSymbol(centerClusters, centerClustersHost, sizeof(double3) * 32));
        KMeans << <dim3(1, 1), dim3(32, 32) >> > (imgDev, w, h, nc);
        ErrorCheck(cudaGetLastError());
        ErrorCheck(cudaMemcpy(imgNew, imgDev, sizeof(uchar4) * w * h, cudaMemcpyDeviceToHost));
        flag = updateClusters(img, imgNew, centerClustersHost, w, h, nc);
        uchar4 * imgTmp = imgNew;
        imgNew = img;
        img = imgTmp;
    }
    ErrorCheck(cudaFree(imgDev));
}

int main() {
    char  inputFileName[256], outFileName[256];
    uint32_t w, h, nc;
    double3 centerClustersHost[32];
    scanf("%s", inputFileName);
    scanf("%s", outFileName);
    FILE *hFile = fopen(inputFileName, "rb");
    fread(&w, sizeof(uint32_t), 1, hFile);
    fread(&h, sizeof(uint32_t), 1, hFile);
    uchar4 *img = (uchar4*)malloc(sizeof(uchar4) * h * w);
    uchar4 *imgNew = (uchar4*)malloc(sizeof(uchar4) * w * h);
    fread(img, sizeof(uchar4), h * w, hFile);
    fclose(hFile);

    scanf("%" SCNu32, &nc);
    for (uint32_t i = 0; i < nc; ++i) {
        int x, y;
        scanf("%" SCNu32 "%" SCNu32, &x, &y);
        centerClustersHost[i].x = img[w * y + x].x;
        centerClustersHost[i].y = img[w * y + x].y;
        centerClustersHost[i].z = img[w * y + x].z;
    }
    cudaEvent_t start, stop;
    float gpuTime = 0.0f;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    KMeans(img, imgNew, centerClustersHost, w, h, nc);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpuTime, start, stop);
    printf("time spent executing by the GPU: %.2f millseconds\n", gpuTime);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    hFile = fopen(outFileName, "wb");
    fwrite(&w, sizeof(uint32_t), 1, hFile);
    fwrite(&h, sizeof(uint32_t), 1, hFile);
    fwrite(img, sizeof(uchar4), w * h, hFile);
    fclose(hFile);
    free(imgNew);
    free(img);
    return 0;
}
