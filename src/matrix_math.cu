#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>

#define CHECK(val)          \
    if (val != cudaSuccess) \
        printf("Err: %s, line: %d\n", cudaGetErrorString(val), __LINE__);

#define THREAD_NUM 256
#define BLOCK_NUM 32

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

    return true;
}

__global__ void matrixMult(float *A, float *B, float *C, int n)
{
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int id = bid * THREAD_NUM + tid;

    int col = id % n;
    int row = id / n;
    if (col < n && row < n)
    {
        float t = 0;
        float y = 0;
        for (int i = 0; i < n; i++)
        {

            y -= A[row * n + i] * B[i * n + col];
            float r = t - y;
            y = r - t + y;
            t = r;
        }
        C[row * n + col] = t;
    }
}

void generateMatrix(float *A, int size)
{
    for (int i = 0; i < size; i++)
    {
        A[i] = i * 0.125465;
    }
}

int main(int argc, char **argv)
{

    if (!initCUDA())
    {
        std::cout << "no cuda device found!" << std::endl;
        return -1;
    }

    std::cout << "cuda device found!" << std::endl;

    int size_n = 10;

    float A[size_n * size_n], B[size_n * size_n], C[size_n * size_n], D[size_n * size_n];

    generateMatrix(A, size_n * size_n);
    generateMatrix(B, size_n * size_n);

    float *ga, *gb, *gc;

    CHECK(cudaMalloc((void **)&ga, size_n * size_n * sizeof(float)));
    CHECK(cudaMalloc((void **)&gb, size_n * size_n * sizeof(float)));
    CHECK(cudaMalloc((void **)&gc, size_n * size_n * sizeof(float)));

    CHECK(cudaMemcpy(ga, A, sizeof(float) * size_n * size_n, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(gb, B, sizeof(float) * size_n * size_n, cudaMemcpyHostToDevice));

    matrixMult<<<BLOCK_NUM, THREAD_NUM, 0>>>(ga, gb, gc, size_n);

    CHECK(cudaMemcpy(C, gc, sizeof(float) * size_n * size_n, cudaMemcpyDeviceToHost));

    // CPU矩阵乘法，存入矩阵d
    for (int i = 0; i < size_n * size_n; i++)
    {
        printf("C[%d] = %f\n", i, C[i]);
    }

    for (int i = 0; i < size_n; i++)
    {
        for (int j = 0; j < size_n; j++)
        {
            double t = 0;

            for (int k = 0; k < size_n; k++)
            {
                t += A[i * size_n + k] * A[k * size_n + j];
            }

            D[i * size_n + j] = t;
        }
    }

    for (int i = 0; i < size_n * size_n; i++)
    {
        printf("C[%d] = %f,D[%d] = %f\n", i, C[i], i, D[i]);
    }

    CHECK(cudaFree(ga));
    CHECK(cudaFree(gb));
    CHECK(cudaFree(gc));

    printf("exit\n");
    return 0;
}