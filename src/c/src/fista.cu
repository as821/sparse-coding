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


void print_stack_trace();
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

#define CHECK_CUBLAS_NORET(func)                                                \
{                                                                               \
    cublasStatus_t status = (func);                                             \
    if (status != CUBLAS_STATUS_SUCCESS) {                                      \
        printf("CUBLAS API failed at line %d with error: (%d) (%s)\n",          \
               __LINE__, status, __FILE__);                                     \
        exit(EXIT_FAILURE);                                                     \
    }                                                                           \
}

__device__ __forceinline__ static void copy_async(float4* gmem_src, float4* smem_dst) {
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst))), "l"(gmem_src), "n"(sizeof(float4)));
}

__device__ __forceinline__ void cp_async_fence() {
    asm volatile("cp.async.commit_group;\n" ::);
}

__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_all;\n" ::);
}

template <int N>
__device__ __forceinline__ void cp_async_wait_n() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}


__device__ __forceinline__ float branchless_relu(float x) {
    return x * (x > 0.0f);
}

template <unsigned int block_sz>
__device__ __forceinline__ void warp_reduce(volatile float* sdata, int tid) {
    if(block_sz >= 64)
        sdata[tid] += sdata[tid + 32];
    if(block_sz >= 32)
        sdata[tid] += sdata[tid + 16];
    if(block_sz >= 16)
        sdata[tid] += sdata[tid + 8];
    if(block_sz >= 8)
        sdata[tid] += sdata[tid + 4];
    if(block_sz >= 4)
        sdata[tid] += sdata[tid + 2];
    if(block_sz >= 2)
        sdata[tid] += sdata[tid + 1];
}

template <unsigned int block_sz, unsigned int n_el_per_thread>
__global__ void y_update(size_t n, float4* __restrict__ Y, float4* __restrict__ z_prev, float alpha_L, float mlt, float* __restrict__ diff_norm, float* __restrict__ prev_z_norm) {
    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;
    
    float thread_local_diff_norm = 0;
    float thread_local_prev_z_norm = 0;

    size_t n_div_4 = n / 4;

    const int n_buffers = 3;        // number of buffers resident in shared memory for either array
    extern __shared__ float4 shmem4[];
    float4* Y_shared_offset = shmem4;
    float4* z_prev_shared_offset = &shmem4[n_buffers * blockDim.x];

    // kick off reads for all buffers except for one
    #pragma unroll
    for(int idx = 0; idx < n_buffers-1; idx++) {
        copy_async(&Y[index + idx * stride], &Y_shared_offset[tid + idx * block_sz]);    
        copy_async(&z_prev[index + idx * stride], &z_prev_shared_offset[tid + idx * block_sz]);
        cp_async_fence();
    }
    int async_buf_idx = 0;
    const int write_idx_step = n_buffers - 1;       // diff between read and write indices

    #pragma unroll
    for(int el_idx = 0; el_idx < n_el_per_thread; el_idx++) {
        int idx = index + el_idx * stride;
        if(idx >= n_div_4)
            break;

        // wait for async for this batch + kick off async for next batch
        cp_async_wait_n<n_buffers-2>();     // only wait for the oldest async copy group (n_buffers-1 in flight at a time, n_buffers-2 after we wait for oldest one here)
        __syncthreads();

        int write_idx = (async_buf_idx + write_idx_step) % n_buffers;            // TODO(as) mod is very slow
        int next_idx = idx + write_idx_step * stride;
        if(next_idx < n_div_4) {
            copy_async(&Y[next_idx], &Y_shared_offset[write_idx * block_sz + tid]);    
            copy_async(&z_prev[next_idx], &z_prev_shared_offset[write_idx * block_sz + tid]);
            cp_async_fence();
        }

        // read from shmem
        float4 Y_vec = Y_shared_offset[async_buf_idx * block_sz + tid];
        float4 z_prev_vec = z_prev_shared_offset[async_buf_idx * block_sz + tid];
        async_buf_idx = (async_buf_idx + 1) % n_buffers;


        // perform computation
        float4 Y_val;
        Y_val.x = branchless_relu(Y_vec.x - alpha_L);
        Y_val.y = branchless_relu(Y_vec.y - alpha_L);
        Y_val.z = branchless_relu(Y_vec.z - alpha_L);
        Y_val.w = branchless_relu(Y_vec.w - alpha_L);

        z_prev[idx] = Y_val;

        float4 diff;
        diff.x = Y_val.x - z_prev_vec.x;
        diff.y = Y_val.y - z_prev_vec.y;
        diff.z = Y_val.z - z_prev_vec.z;
        diff.w = Y_val.w - z_prev_vec.w;
        
        thread_local_prev_z_norm += z_prev_vec.x * z_prev_vec.x + z_prev_vec.y * z_prev_vec.y + z_prev_vec.z * z_prev_vec.z + z_prev_vec.w * z_prev_vec.w;

        thread_local_diff_norm += diff.x * diff.x + diff.y * diff.y + diff.z * diff.z + diff.w * diff.w;

        Y_val.x += mlt * diff.x;
        Y_val.y += mlt * diff.y;
        Y_val.z += mlt * diff.z;
        Y_val.w += mlt * diff.w;

        Y[idx] = Y_val;
    }

    {
        // tree-based reduction of thread-local norm values for all threads in the block
        // https://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/reduction/doc/reduction.pdf
        extern __shared__ float shmem[];
        float* shared_diff_norm = shmem;
        float* shared_prev_z_norm = &shmem[blockDim.x];

        shared_diff_norm[tid] = thread_local_diff_norm;
        shared_prev_z_norm[tid] = thread_local_prev_z_norm;
        __syncthreads();

        if(block_sz >= 512) {
            if(tid < 256) {
                shared_diff_norm[tid] += shared_diff_norm[tid + 256];
                shared_prev_z_norm[tid] += shared_prev_z_norm[tid + 256];
            }
            __syncthreads();
        }
        if(block_sz >= 256) {
            if(tid < 128) {
                shared_diff_norm[tid] += shared_diff_norm[tid + 128];
                shared_prev_z_norm[tid] += shared_prev_z_norm[tid + 128];
            }
            __syncthreads();
        }
        if(block_sz >= 128) {
            if(tid < 64) {
                shared_diff_norm[tid] += shared_diff_norm[tid + 64];
                shared_prev_z_norm[tid] += shared_prev_z_norm[tid + 64];
            }
            __syncthreads();
        }
        if(tid < 32) {
            warp_reduce<block_sz>(shared_diff_norm, tid);
            warp_reduce<block_sz>(shared_prev_z_norm, tid);
        }

        if (tid == 0) {
            atomicAdd(diff_norm, shared_diff_norm[0]);
            atomicAdd(prev_z_norm, shared_prev_z_norm[0]);
        }
    }
}

void print_norm_host(float* arr, size_t sz, const char* str) {
    double norm = 0;
    for(int idx = 0; idx < sz; idx++) {
        norm += (double)arr[idx] * (double)arr[idx];
    }
    norm = sqrt(norm);
    printf("%s: %f\n", str, norm);
}

void print_norm(float* arr_dev, size_t sz, const char* str) {
    float* arr = (float*)malloc(sz * sizeof(float));
    CHECK(arr);
    CHECK_CUDA_NORET(cudaMemcpy((void*)arr, arr_dev, sz * sizeof(float), cudaMemcpyDeviceToHost))
    print_norm_host(arr, sz, str);
    free(arr);
}

void cuda_log_time_diff(char* msg, cudaEvent_t* start, cudaEvent_t* stop) {
    float milli = 0;
    cudaEventElapsedTime(&milli, *start, *stop);
    milli /= 1000;      // ms -> s
    printf("%s: %f\n", msg, milli);
}

void log_time_diff(char* msg, struct timeval* start, struct timeval* stop) {
    double start_ms = (((double)start->tv_sec)*1000)+(((double)start->tv_usec)/1000);
    double stop_ms = (((double)stop->tv_sec)*1000)+(((double)stop->tv_usec)/1000);
    double diff_in_sec = (stop_ms - start_ms)/1000;
    printf("%s: %f\n", msg, diff_in_sec);
}


extern "C" {
int fista(float* __restrict__ X_host, float* __restrict__ basis_host, float* __restrict__ Z_host, int n_samples, int inp_dim, int dict_sz, float lr, float alpha_L, int n_iter, float converge_thresh) {
    CHECK(X_host);
    CHECK(basis_host);
    CHECK(Z_host);

    struct timeval actual_start, handle_time, init, exec;
    gettimeofday(&actual_start, NULL);


    // X: n_samples x inp_dim
    // basis: inp_dim x dict_sz
    // Z: n_samples x dict_sz

    // TODO(as): (very) slow the first time it is called in a process...
    cublasHandle_t handle;
    cublasCreate(&handle);

    gettimeofday(&handle_time, NULL);


    // TODO(as): bunch of faster + less precise BLAS options here https://docs.nvidia.com/cuda/cublas/#cublasoperation-t
    // TODO(as): CUTLASS? https://github.com/NVIDIA/cutlass/blob/main/examples/45_dual_gemm/dual_gemm.cu
    cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;


    // TODO(as): page locking + async transfers. ways to move these calls outside of this function so not called on each iteration?
    float *X, *basis;
    size_t x_n_el = n_samples * inp_dim;
    size_t x_sz = x_n_el * sizeof(float);
    size_t basis_sz = inp_dim * dict_sz * sizeof(float);
    CHECK_CUDA_NORET(cudaMalloc((void**)&X, x_sz))
    CHECK_CUDA_NORET(cudaMalloc((void**)&basis, basis_sz))
    CHECK_CUDA_NORET(cudaMemcpy((void*)X, X_host, x_sz, cudaMemcpyHostToDevice))
    CHECK_CUDA_NORET(cudaMemcpy((void*)basis, basis_host, basis_sz, cudaMemcpyHostToDevice))

    float *residual, *z_prev, *Y;
    size_t z_n_el = dict_sz * n_samples;
    size_t z_sz = z_n_el * sizeof(float);
    CHECK_CUDA_NORET(cudaMalloc((void**)&residual, x_sz))
    CHECK_CUDA_NORET(cudaMalloc((void**)&z_prev, z_sz))
    CHECK_CUDA_NORET(cudaMalloc((void**)&Y, z_sz))
    CHECK_CUDA_NORET(cudaMemset(z_prev, 0, z_sz))
    CHECK_CUDA_NORET(cudaMemset(Y, 0, z_sz))


    CHECK(z_n_el % 4 == 0);         // assumed by kernel format

    float norms_host[2];
    float* norms;
    size_t norm_sz = 2 * sizeof(float);
    CHECK_CUDA_NORET(cudaMalloc((void**)&norms, norm_sz))

    gettimeofday(&init, NULL);

    float tk = 1, tk_prev = 1;
    int itr;
    for(itr = 0; itr < n_iter; itr++) {

        cudaEvent_t start, blas, k_start, k_exec, k_end;
        cudaEventCreate(&start);
        cudaEventCreate(&blas);
        cudaEventCreate(&k_start);
        cudaEventCreate(&k_exec);
        cudaEventCreate(&k_end);
        cudaEventRecord(start);

        // residual = x - (z @ basis.T)
        CHECK_CUDA_NORET(cudaMemcpy((void*)residual, X, x_sz, cudaMemcpyDeviceToDevice))
        {
            // cublas assumes column-major but we have row major
            // https://i.sstatic.net/IvZPe.png
            float alpha = -1.0f;
            float beta = 1.0f;
            CHECK_CUBLAS_NORET(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, inp_dim, n_samples, dict_sz, &alpha, basis, CUDA_R_32F, dict_sz, Y, CUDA_R_32F, dict_sz, &beta, residual, CUDA_R_32F, inp_dim, compute_type, CUBLAS_GEMM_DEFAULT));    
        }

        // mm = residual @ basis
        // z += lr * mm
        {
            // cublas assumes column-major but we have row major
            // https://i.sstatic.net/IvZPe.png
            float beta = 1.0f;
            CHECK_CUBLAS_NORET(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, dict_sz, n_samples, inp_dim, &lr, basis, CUDA_R_32F, dict_sz, residual, CUDA_R_32F, inp_dim, &beta, Y, CUDA_R_32F, dict_sz, compute_type, CUBLAS_GEMM_DEFAULT));
        }

        cudaEventRecord(blas);

        // Y-update multiplier
        tk_prev = tk;
        tk = (1 + sqrtf(1 + 4 * tk * tk)) / 2;
        float mlt = (tk_prev - 1) / tk;
        CHECK_CUDA_NORET(cudaMemset(norms, 0, norm_sz))
        cudaEventRecord(k_start);
        
        const int block_sz = 32;
        // int n_blocks = (ceil(z_n_el / 4) + block_sz - 1) / block_sz;       // ceil(z_n_el / block_sz)
        // int n_blocks = 8192;
        
        const int n_el_per_thread = 64;
        int n_blocks = (int)ceil((float)z_n_el / (float)(n_el_per_thread * block_sz));
        printf("nblocks: %d\n", n_blocks);

        // int smem_sz = 2 * block_sz * sizeof(float);
        int smem_sz = 6 * sizeof(float4) * block_sz;
        y_update<block_sz, n_el_per_thread><<<n_blocks, block_sz, smem_sz>>>(z_n_el, (float4*)Y, (float4*)z_prev, alpha_L, mlt, norms, &norms[1]);
        cudaEventRecord(k_exec);

        CHECK_CUDA_NORET(cudaMemcpy((void*)norms_host, norms, norm_sz, cudaMemcpyDeviceToHost))

        // Frobenius norm can be defined as the L2 norm of the flattened matrix
        float diff_norm = norms_host[0];
        float prev_z_norm = norms_host[1];
        float norm_ratio = diff_norm / prev_z_norm;
        norm_ratio = sqrtf(norm_ratio);         // equivalent to sqrtf(diff_norm) / sqrtf(prev_z_norm)
        cudaEventRecord(k_end);


        // printf("%d: %f %f\n", itr, sqrtf(diff_norm), sqrtf(prev_z_norm));
        
        printf("\33[2K\r%d / %d", itr, n_iter);
        fflush(stdout);



        cudaEventSynchronize(k_end);
        cuda_log_time_diff("\n\tblas", &start, &blas);
        cuda_log_time_diff("\tk_start", &blas, &k_start);
        cuda_log_time_diff("\tk_exec", &k_start, &k_exec);
        cuda_log_time_diff("\tk_end", &k_exec, &k_end);
        float milli = 0;
        cudaEventElapsedTime(&milli, k_start, k_exec);
        printf("\tbandwidth: %f (GB/s)\n", z_sz * 4 / milli / 1e6);     // 2 read + 2 write per iteration   

        if(itr != 0 && norm_ratio < converge_thresh)
            break;
    }
    // printf("\n");

    // memcpy(Z, z_prev, dict_sz * n_samples * sizeof(float));
    CHECK_CUDA_NORET(cudaMemcpy((void*)Z_host, z_prev, z_sz, cudaMemcpyDeviceToHost))


    CHECK_CUDA_NORET(cudaFree(residual))
    CHECK_CUDA_NORET(cudaFree(z_prev))
    CHECK_CUDA_NORET(cudaFree(norms))
    CHECK_CUDA_NORET(cudaFree(Y))
    CHECK_CUDA_NORET(cudaFree(X))
    CHECK_CUDA_NORET(cudaFree(basis))

    cublasDestroy(handle);


    gettimeofday(&exec, NULL);


    log_time_diff("\n\n\thandle", &actual_start, &handle_time);
    log_time_diff("\tinit", &handle_time, &init);
    log_time_diff("\texec", &init, &exec);
    return itr;
}
}


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

