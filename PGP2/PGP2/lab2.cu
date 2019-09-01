/*#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>

#define ErrorCheck(ans) { CheckFun((ans), __FILE__, __LINE__); }
#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

inline void CheckFun(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "ERROR: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(0);
    }
}

texture<uchar4, cudaTextureType2D, cudaReadModeElementType> texRef;

__device__ inline int getColorComponentByIDx(int x, int y) {
    if (blockIdx.x == 0) {
        return tex2D(texRef, x, y).x;
    }
    if (blockIdx.x == 1) {
        return tex2D(texRef, x, y).y;
    }
    if (blockIdx.x == 2) {
        return tex2D(texRef, x, y).z;
    }
    return 0;
}
__device__ inline void setColorComponentByIDx(uchar4* res, int index, int median) {
    if (blockIdx.x == 0) {
        res[index].x = median;
    }
    if (blockIdx.x == 1) {
        res[index].y = median;
    }
    if (blockIdx.x == 2) {
        res[index].z = median;
    }
    return;
}

__device__   void initHistogram(int histogram[], int &mid, int idy, int h, int w, int r) {
    int hTop = MAX(idy - r, 0);
    int hBot = MIN(idy + r, h - 1);
    int wRight = MIN(r, w - 1);
    for (int i = 0; i < 256; ++i) {
        histogram[i] = 0;
    }
    mid = (hBot - hTop + 1) * (wRight + 1) / 2;
    for (int i = hTop; i <= hBot; ++i) {
        for (int j = 0; j <= wRight; ++j) {
            histogram[getColorComponentByIDx(j, i)] += 1;
        }
    }
    return;
}

__device__  inline int findMedian(int histogram[], int mid) {
    int i = 0, count = 0;
    for (; count <= mid; ++i) {
        count += histogram[i];
    }
    return i - 1;
}

__device__   void updateHistogram(int histogram[], int &mid, int idy, int idx, int h, int w, int r) {
    int hTop = MAX(idy - r, 0);
    int hBot = MIN(idy + r, h - 1);
    int wRight = MIN(idx + r, w - 1);
    int wLeft = MAX(idx - r, 0);
    mid = (hBot - hTop + 1) * (wRight - wLeft + 1) / 2;
    for (int i = hTop; i <= hBot; ++i) {
        if (idx - r - 1 >= 0) {
            histogram[getColorComponentByIDx(wLeft - 1, i)] -= 1;
        }
        if (idx + r <= w - 1) {
            histogram[getColorComponentByIDx(wRight, i)] += 1;
        }
    }
    return;
}

__global__ void MedianFilter(uchar4 *res, int h, int w, int r) {
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    int offsety = blockDim.y * gridDim.y;
    for (int i = idy; i < h; i += offsety) {
        int histogram[256];
        int mid;
        initHistogram(histogram, mid, i, h, w, r);
        setColorComponentByIDx(res, i * w, findMedian(histogram, mid));
        for (int j = 1; j < w; ++j) {
            updateHistogram(histogram, mid, i, j, h, w, r);
            setColorComponentByIDx(res, i * w + j, findMedian(histogram, mid));
        }
    }
    return;
}


int main() {
    char  inputFileName[256], outFileName[256];
    int w, h, r = 1;
    scanf("%s", inputFileName);
    scanf("%s", outFileName);
    scanf("%d", &r);
    FILE *hFile = fopen(inputFileName, "rb");
    fread(&w, sizeof(int), 1, hFile);
    fread(&h, sizeof(int), 1, hFile);
    uchar4 *img = (uchar4*)malloc(sizeof(uchar4) * h * w);
    fread(img, sizeof(uchar4), h * w, hFile);
    fclose(hFile);
   
    cudaArray *dev_arr;
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();
    ErrorCheck(cudaMallocArray(&dev_arr, &channelDesc, w, h));
    cudaMemcpyToArray(dev_arr, 0, 0, img, sizeof(uchar4) * w *h, cudaMemcpyHostToDevice);
    texRef.addressMode[0] = cudaAddressModeClamp;
    texRef.addressMode[1] = cudaAddressModeClamp;
    texRef.channelDesc = channelDesc;
    texRef.filterMode = cudaFilterModePoint;
    texRef.normalized = false;

    ErrorCheck(cudaBindTextureToArray(texRef, dev_arr, channelDesc));
    uchar4 *dev_img;
    cudaMalloc(&dev_img, sizeof(uchar4) * w * h);
    MedianFilter <<<dim3(3, 64,1), dim3(1,64,1)>>> (dev_img, h, w, r);
    ErrorCheck(cudaGetLastError());

    ErrorCheck(cudaMemcpy(img, dev_img, sizeof(uchar4) * w * h, cudaMemcpyDeviceToHost));

    ErrorCheck(cudaUnbindTexture(texRef));
    ErrorCheck(cudaFreeArray(dev_arr));
    ErrorCheck(cudaFree(dev_img));

    hFile = fopen(outFileName, "wb");
    fwrite(&w, sizeof(int), 1, hFile);
    fwrite(&h, sizeof(int), 1, hFile);
    fwrite(img, sizeof(uchar4), w * h, hFile);
    fclose(hFile);

    free(img);
    return 0;
} */