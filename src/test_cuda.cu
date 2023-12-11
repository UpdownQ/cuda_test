#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>

#define CHECK(val)          \
    if (val != cudaSuccess) \
        printf("Err: %s, line: %d\n", cudaGetErrorString(val), __LINE__);

#define DATA_SIZE 100000
#define THREAD_NUM 256
#define BLOCK_NUM 32

void printDeviceProp(const cudaDeviceProp &prop)
{
    printf("Device Name : %s.\n", prop.name);
    printf("totalGlobalMem : %d.\n", prop.totalGlobalMem);
    printf("sharedMemPerBlock : %d.\n", prop.sharedMemPerBlock);
    printf("regsPerBlock : %d.\n", prop.regsPerBlock);
    printf("warpSize : %d.\n", prop.warpSize);
    printf("memPitch : %d.\n", prop.memPitch);
    printf("maxThreadsPerBlock : %d.\n", prop.maxThreadsPerBlock);
    printf("maxThreadsDim[0 - 2] : %d %d %d.\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("maxGridSize[0 - 2] : %d %d %d.\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("totalConstMem : %d.\n", prop.totalConstMem);
    printf("major.minor : %d.%d.\n", prop.major, prop.minor);
    printf("clockRate : %d.\n", prop.clockRate);
    printf("textureAlignment : %d.\n", prop.textureAlignment);
    printf("deviceOverlap : %d.\n", prop.deviceOverlap);
    printf("multiProcessorCount : %d.\n", prop.multiProcessorCount);
}

bool initCUDA()
{
    int count;
    // 取得支持Cuda的装置的数目
    cudaGetDeviceCount(&count);
    // 没有符合的硬件
    if (count == 0)
    {
        fprintf(stderr, "There is no device.\n");
        return false;
    }

    cudaDeviceProp devProp;
    CHECK(cudaGetDeviceProperties(&devProp, 0));

    printDeviceProp(devProp);

    cudaSetDevice(0);

    return true;
}

// 生成随机数
void generateRandom(int *data, int size)

{
    for (int i = 0; i < size; i++)
    {
        data[i] = i;
    }
}

__global__ void sumOfData(int *data, int *result)
{
    extern __shared__ int shared[];

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int i;
    for (i = bid * THREAD_NUM + tid; i < DATA_SIZE; i += THREAD_NUM * BLOCK_NUM)
    {
        shared[tid] += data[i];
    }

    __syncthreads();
    // 树状加法
    int add_id = 1;
    while (tid + add_id < THREAD_NUM)
    {
        shared[tid] += shared[tid + add_id];
        add_id *= 2;
        __syncthreads(); // 很重要
    }
    result[bid] = shared[0];
}

int main(int *args, char **argv)
{

    if (!initCUDA())
    {
        return 0;
    }
    printf("CUDA initialized.\n");
    int sum = 0;
    int data[DATA_SIZE];
    int result[THREAD_NUM * BLOCK_NUM];
    generateRandom(data, DATA_SIZE);

    int *gdata, *gresult;

    CHECK(cudaMalloc((void **)&gdata, sizeof(int) * DATA_SIZE));
    CHECK(cudaMalloc((void **)&gresult, sizeof(int) * BLOCK_NUM));

    CHECK(cudaMemcpy(gdata, data, sizeof(int) * DATA_SIZE, cudaMemcpyHostToDevice));

    sumOfData<<<BLOCK_NUM, THREAD_NUM, BLOCK_NUM * sizeof(int)>>>(gdata, gresult);

    CHECK(cudaMemcpy(result, gresult, sizeof(int) * BLOCK_NUM, cudaMemcpyDeviceToHost));

    for (int i = 0; i < BLOCK_NUM; i++)
    {
        sum += result[i];
    }

    printf("sum = %d\n", sum);

    CHECK(cudaFree(gdata));
    CHECK(cudaFree(gresult));

    return 0;
}