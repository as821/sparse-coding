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

__global__ void y_update(size_t n, float* Y, float* z_prev, float alpha_L, float mlt, float* diff_norm, float* prev_z_norm) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    float thread_local_diff_norm = 0;
    float thread_local_prev_z_norm = 0;
    
    for(int idx = index; idx < n; idx += stride) {
        Y[idx] = Y[idx] < alpha_L ? 0 : Y[idx] - alpha_L;       // TODO(as) make sure nvcc uses a vector blend rather than branch here?
        float diff = Y[idx] - z_prev[idx];
        
        thread_local_diff_norm += diff * diff;
        thread_local_prev_z_norm += z_prev[idx] * z_prev[idx];
        
        z_prev[idx] = Y[idx];
        Y[idx] += mlt * diff;
    }

    // synchronize access to shared result variables
    atomicAdd(diff_norm, thread_local_diff_norm);
    atomicAdd(prev_z_norm, thread_local_prev_z_norm);
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

extern "C" {
void fista(float* __restrict__ X_host, float* __restrict__ basis_host, float* __restrict__ Z_host, int n_samples, int inp_dim, int dict_sz, float lr, float alpha_L, int n_iter, float converge_thresh) {
    CHECK(X_host);
    CHECK(basis_host);
    CHECK(Z_host);

    // X: n_samples x inp_dim
    // basis: inp_dim x dict_sz
    // Z: n_samples x dict_sz

    cublasHandle_t handle;
    cublasCreate(&handle);

    cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F_PEDANTIC;

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

    float norms_host[2];
    float* norms;
    size_t norm_sz = 2 * sizeof(float);
    CHECK_CUDA_NORET(cudaMalloc((void**)&norms, norm_sz))

    float tk = 1, tk_prev = 1;
    for(int itr = 0; itr < n_iter; itr++) {
        
        // {
        //     float* Y_host = (float*)malloc(z_sz);
        //     CHECK_CUDA_NORET(cudaMemcpy((void*)Y_host, Y, z_sz, cudaMemcpyDeviceToHost))
        //     for(int idx = 0; idx < z_n_el; idx++) {
        //         printf("\tY pre (%d): %f\n", idx, Y_host[idx]);
        //     }
        // }
        
        
        // residual = x - (z @ basis.T)
        // cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n_samples, inp_dim, dict_sz, -1.0f, Y, dict_sz, basis, dict_sz, 1.0f, residual, inp_dim);
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
        // cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n_samples, dict_sz, inp_dim, lr, residual, inp_dim, basis, dict_sz, 1.0f, Y, dict_sz);
        {
            // cublas assumes column-major but we have row major
            // https://i.sstatic.net/IvZPe.png
            float beta = 1.0f;
            CHECK_CUBLAS_NORET(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, dict_sz, n_samples, inp_dim, &lr, basis, CUDA_R_32F, dict_sz, residual, CUDA_R_32F, inp_dim, &beta, Y, CUDA_R_32F, dict_sz, compute_type, CUBLAS_GEMM_DEFAULT));
        }



        // float* Y_host = (float*)malloc(z_sz);
        // CHECK_CUDA_NORET(cudaMemcpy((void*)Y_host, Y, z_sz, cudaMemcpyDeviceToHost))
        // for(int idx = 0; idx < z_n_el; idx++) {
        //     printf("\tY (%d): %f\n", idx, Y_host[idx]);
        // }

        // float* res_host = (float*)malloc(x_sz);
        // CHECK_CUDA_NORET(cudaMemcpy((void*)res_host, residual, x_sz, cudaMemcpyDeviceToHost))
        // for(int idx = 0; idx < inp_dim * n_samples; idx++) {
        //     printf("\tres (%d): %f\n", idx, res_host[idx]);
        // }


        // Y-update multiplier
        tk_prev = tk;
        tk = (1 + sqrtf(1 + 4 * tk * tk)) / 2;
        float mlt = (tk_prev - 1) / tk;
        CHECK_CUDA_NORET(cudaMemset(norms, 0, norm_sz))
        
        int block_sz = 256;         // TODO(as) look up threads per block for 4090?
        int n_blocks = (z_n_el + block_sz - 1) / block_sz;       // ceil(z_n_el / block_sz)
        y_update<<<n_blocks, block_sz>>>(z_n_el, Y, z_prev, alpha_L, mlt, norms, &norms[1]);
        CHECK_CUDA_NORET(cudaDeviceSynchronize());
        CHECK_CUDA_NORET(cudaMemcpy((void*)norms_host, norms, norm_sz, cudaMemcpyDeviceToHost))

        // Frobenius norm can be defined as the L2 norm of the flattened matrix
        float diff_norm = norms_host[0];
        float prev_z_norm = norms_host[1];
        float norm_ratio = diff_norm / prev_z_norm;
        norm_ratio = sqrtf(norm_ratio);         // equivalent to sqrtf(diff_norm) / sqrtf(prev_z_norm)


        // printf("%d: %f %f\n", itr, sqrtf(diff_norm), sqrtf(prev_z_norm));
        printf("\33[2K\r%d / %d", itr, n_iter);
        fflush(stdout);



        if(itr != 0 && norm_ratio < converge_thresh)
            break;
    }
    printf("\n");

    // memcpy(Z, z_prev, dict_sz * n_samples * sizeof(float));
    CHECK_CUDA_NORET(cudaMemcpy((void*)Z_host, z_prev, z_sz, cudaMemcpyDeviceToHost))


    CHECK_CUDA_NORET(cudaFree(residual))
    CHECK_CUDA_NORET(cudaFree(z_prev))
    CHECK_CUDA_NORET(cudaFree(Y))
    CHECK_CUDA_NORET(cudaFree(X))
    CHECK_CUDA_NORET(cudaFree(basis))

    cublasDestroy(handle);
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


