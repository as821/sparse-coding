#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <execinfo.h>
#include <sys/time.h>
#include <stdbool.h>
#include <math.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_pipeline.h>


#include <random>
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>


// Benchmarking setup largely taken from https://leimao.github.io/blog/CUDA-Shared-Memory-Swizzling/


void print_stack_trace() {
    void *array[100];
    size_t size;
    char **strings;
    size_t i;

    size = backtrace(array, 100);
    strings = backtrace_symbols(array, size);

    printf("Stack trace:\n");
    for (i = 0; i < size; i++) {
        printf("\t%s\n", strings[i]);
    }

    free(strings);
}

#define CHECK(x)                                                                                    \
{                                                                                                   \
    if(!(x)) {                                                                                      \
        printf("ERROR (line %d, file:%s) (%d): %s\n", __LINE__, __FILE__, errno, strerror(errno));  \
        print_stack_trace();                                                                        \
        exit(EXIT_FAILURE);                                                                         \
    }                                                                                               \
}

#define CHECK_CUDA_NORET(func)                                                  \
{                                                                               \
    cudaError_t status = (func);                                                \
    if (status != cudaSuccess) {                                                \
        printf("CUDA API failed at line %d with error: %s (%d) (%s)\n",         \
               __LINE__, cudaGetErrorString(status), status, __FILE__);         \
        exit(EXIT_FAILURE);                                                     \
    }                                                                           \
}

#define CHECK_LAST_CUDA_ERROR() check_last(__FILE__, __LINE__)
void check_last(char const* file, int line) {
    cudaError_t const err{cudaGetLastError()};
    if (err != cudaSuccess) {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}


bool validate(float* src, float* dst, size_t n_el) {
    float* src_host = (float*) malloc(n_el * sizeof(float));
    CHECK(src_host)
    float* dst_host = (float*) malloc(n_el * sizeof(float));
    CHECK(dst_host)
    
    CHECK_CUDA_NORET(cudaMemcpy((void*)src_host, src, n_el * sizeof(float), cudaMemcpyDeviceToHost))
    CHECK_CUDA_NORET(cudaMemcpy((void*)dst_host, dst, n_el * sizeof(float), cudaMemcpyDeviceToHost))


    for(int idx = 0; idx < n_el; idx++) {
        if(src_host[idx] != dst_host[idx]) {
            printf("Failure index %d: %f != %f\n", idx, src_host[idx], dst_host[idx]);
            free(src_host);
            free(dst_host);
            return false;
        }
    }
    free(src_host);
    free(dst_host);
    return true;
}

float eval(void (*kernel_func)(float*, float*, size_t), float* src, float* dst, size_t n_el) {
    size_t num_warmups = 10;
    size_t num_repeats = 50;

    cudaEvent_t start, stop;
    float time;

    for (size_t i = 0; i < num_warmups; ++i) {
        kernel_func(src, dst, n_el);
    }

    CHECK_CUDA_NORET(cudaEventCreate(&start));
    CHECK_CUDA_NORET(cudaEventCreate(&stop));
    CHECK_CUDA_NORET(cudaDeviceSynchronize());

    CHECK_CUDA_NORET(cudaEventRecord(start));
    for (size_t i = 0; i < num_repeats; ++i) {
        kernel_func(src, dst, n_el);
    }
    CHECK_CUDA_NORET(cudaEventRecord(stop));
    CHECK_CUDA_NORET(cudaEventSynchronize(stop));
    CHECK_LAST_CUDA_ERROR();

    CHECK_CUDA_NORET(cudaEventElapsedTime(&time, start, stop));
    
    CHECK_CUDA_NORET(cudaEventDestroy(start));
    CHECK_CUDA_NORET(cudaEventDestroy(stop));

    float perf = time / num_repeats;

    if(!validate(src, dst, n_el)) 
        printf("FAILURE!!!\n");
    else
        printf("\tPerformance: %f\n", perf);

    return perf;
}


void launch_memcpy(float* src, float* dst, size_t n_el) {
    CHECK_CUDA_NORET(cudaMemcpy((void*)dst, src, n_el * sizeof(float), cudaMemcpyDeviceToDevice))
}


__global__ void copy_1(float* src, float* dst, size_t n_el) {
    int tid = threadIdx.x;
    int step_sz = blockDim.x * gridDim.x;
    int index = blockIdx.x * blockDim.x + tid;

    for(int idx = index; idx < n_el; idx += step_sz) {
        dst[idx] = src[idx];
    }
}

void launch_copy_1(float* src, float* dst, size_t n_el) {
    size_t block_sz = 1024;
    size_t el_per_thread = 32;
    size_t grid_sz = (n_el + (block_sz * el_per_thread) - 1) / (block_sz * el_per_thread);

    copy_1<<<grid_sz, block_sz>>>(src, dst, n_el);
}
__global__ void copy_2(float* src, float* dst, size_t n_el) {
    int tid = threadIdx.x;
    int step_sz = blockDim.x * gridDim.x;
    int index = blockIdx.x * blockDim.x + tid;

    int idx = index * 4;
    for(; idx < n_el; idx += step_sz * 4) {
        int idx_by_4 = idx / 4;
        ((float4*)dst)[idx_by_4] = ((float4*)src)[idx_by_4];
    }
    for(; idx < n_el; idx++) {
        dst[idx] = src[idx];
    }
}

void launch_copy_2(float* src, float* dst, size_t n_el) {
    size_t block_sz = 1024;
    size_t el_per_thread = 32;
    size_t grid_sz = (n_el + (block_sz * el_per_thread) - 1) / (block_sz * el_per_thread);
    copy_2<<<grid_sz, block_sz>>>(src, dst, n_el);
}


__global__ void copy_3(float* __restrict__ src, float* __restrict__ dst, size_t n_el, size_t el_per_thread) {
    int tid = threadIdx.x;
    int step_sz = blockDim.x * gridDim.x;
    int index = blockIdx.x * blockDim.x + tid;

    extern __shared__ float shmem[];

    int thread_cnt = 0;
    for(int idx = index * 4; idx < n_el; idx += step_sz * 4) {
        int idx_by_4 = idx / 4;
        __pipeline_memcpy_async(&((float4*)shmem)[tid + blockDim.x * thread_cnt], &((float4*)src)[idx_by_4], sizeof(float4));
        thread_cnt++;
    }

    __pipeline_commit();
    __pipeline_wait_prior(0);

    thread_cnt = 0;
    int idx = index * 4;
    for(; idx < n_el; idx += step_sz * 4) {
        int idx_by_4 = idx / 4;
        ((float4*)dst)[idx_by_4] = ((float4*)shmem)[tid + blockDim.x * thread_cnt];
        thread_cnt++;
    }

    for(; idx < n_el; idx++) {
        dst[idx] = src[idx];
    }
}

void launch_copy_3(float* src, float* dst, size_t n_el) {
    size_t block_sz = 32;
    size_t el_per_thread = 64;
    size_t grid_sz = (n_el + (block_sz * el_per_thread) - 1) / (block_sz * el_per_thread);
    size_t shared_mem_sz = block_sz * sizeof(float4) * el_per_thread;

    copy_3<<<grid_sz, block_sz, shared_mem_sz>>>(src, dst, n_el, el_per_thread);
}






int main() {
    // initialize random input buffer and move to device
    printf("Setting up...\n");
    size_t n_el = 1e9;
    float* host_buf = (float*) malloc(n_el * sizeof(float));
    CHECK(host_buf);

    float val = 8192;
    std::mt19937 gen{0};
    std::uniform_real_distribution<float> uniform_dist(-1 * val, val);
    for (size_t i = 0; i < n_el; i++) {
        host_buf[i] = uniform_dist(gen);
    }

    float* device_buf = nullptr, *dest_buf = nullptr;
    CHECK_CUDA_NORET(cudaMalloc((void**)&device_buf, n_el * sizeof(float)))
    CHECK_CUDA_NORET(cudaMalloc((void**)&dest_buf, n_el * sizeof(float)))
    CHECK_CUDA_NORET(cudaMemcpy((void*)device_buf, host_buf, n_el * sizeof(float), cudaMemcpyHostToDevice))

    // printf("memcpy: \n");
    // eval(launch_memcpy, device_buf, dest_buf, n_el);

    // printf("copy 1: \n");
    // eval(launch_copy_1, device_buf, dest_buf, n_el);

    // printf("copy 2: \n");
    // eval(launch_copy_2, device_buf, dest_buf, n_el);

    printf("copy 3: \n");
    eval(launch_copy_3, device_buf, dest_buf, n_el);


    CHECK_CUDA_NORET(cudaFree(device_buf))
    CHECK_CUDA_NORET(cudaFree(dest_buf))
    free(host_buf);
}




