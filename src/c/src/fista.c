
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <execinfo.h>
#include <sys/time.h>
#include <stdbool.h>

#include <math.h>
#include <cblas.h>

#include <immintrin.h>


void print_stack_trace();

// 5950x
//  - 3.4 base (4.9 boost)
//  - L1d: 512K (32k / core)
//  - L2: 8 MB (512K / core)
//  - L3: 64 MB

// https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html
// https://siboehm.com/articles/22/Fast-MMM-on-CPU
// https://www.agner.org/optimize/optimizing_cpp.pdf
// https://www.netlib.org/blas/
// https://www.akkadia.org/drepper/cpumemory.pdf


#define ALIGNMENT 64            // cache line size for Zen 3, greater than 32 bytes required for aligned AVX256 load/store

#define SPARSITY_DEBUG false
#define DEBUG false


#define CHECK(x)                                                                                    \
{                                                                                                   \
    if(!(x)) {                                                                                      \
        printf("ERROR (line %d, file:%s) (%d): %s\n", __LINE__, __FILE__, errno, strerror(errno));  \
        print_stack_trace();                                                                        \
        exit(EXIT_FAILURE);                                                                         \
    }                                                                                               \
}



void log_time_diff(char* msg, struct timeval* start, struct timeval* stop) {
    double start_ms = (((double)start->tv_sec)*1000)+(((double)start->tv_usec)/1000);
    double stop_ms = (((double)stop->tv_sec)*1000)+(((double)stop->tv_usec)/1000);
    double diff_in_sec = (stop_ms - start_ms)/1000;

    if(DEBUG)
        printf("%s: %f\n", msg, diff_in_sec);
}


float n_nonzero_elements(float* m, int sz) {
    int cnt = 0;
    for(int idx = 0; idx < sz; idx++) {
        if(fabs(m[idx]) > 0.00001) 
            cnt++;
    }
    return ((float)cnt) / ((float)sz);
}


void print_m256(__m256 vec) {
    float values[8];
    _mm256_storeu_ps(values, vec);
    printf("[ ");
    for(int i = 0; i < 8; i++) {
        printf("%f ", values[i]);
    }
    printf("]\n");
}


float horizontal_add(__m256 v) {
    // add the high and low 128 bits
    __m128 sum128 = _mm_add_ps(_mm256_extractf128_ps(v, 1), _mm256_castps256_ps128(v));
    
    // horizontal add the 4 floats in the 128-bit vector
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    return _mm_cvtss_f32(sum128);
}


void matrix_transpose(float* src, float* dst, int M, int N) {
    // Transpose the given M x N matrix. Cache locality will be poor on either src or dst due to the nature of the problem
    for(int idx = 0; idx < M; idx++) {
        for(int jdx = 0; jdx < N; jdx++) {
            dst[jdx * M + idx] = src[idx * N + jdx];
        }
    }
}

void print_norm(float* arr, size_t sz, const char* str) {
    double norm = 0;
    for(int idx = 0; idx < sz; idx++) {
        norm += (double)arr[idx] * (double)arr[idx];
    }
    norm = sqrt(norm);
    printf("%s: %f\n", str, norm);
}


int fista(float* __restrict__ X, float* __restrict__ basis, float* __restrict__ Z, int n_samples, int inp_dim, int dict_sz, float L_inv, float alpha_L, int n_iter, float converge_thresh) {
    CHECK(X);
    CHECK(basis);
    CHECK(Z);

    // X: n_samples x inp_dim
    // basis: inp_dim x dict_sz
    // Z: n_samples x dict_sz

    float* residual = (float*) aligned_alloc(ALIGNMENT, inp_dim * n_samples * sizeof(float));
    CHECK(residual);

    float* z_prev = (float*) aligned_alloc(ALIGNMENT, dict_sz * n_samples * sizeof(float));
    memset(z_prev, 0, dict_sz * n_samples * sizeof(float));
    CHECK(z_prev);

    float* Y = (float*) aligned_alloc(ALIGNMENT, dict_sz * n_samples * sizeof(float));
    CHECK(Y);
    memset(Y, 0, dict_sz * n_samples * sizeof(float));

    // assumed by vectorization
    // CHECK(n_samples % 8 == 0);

    float tk = 1, tk_prev = 1;
    float x_nz, basis_nz, y_nz_pre, res_nz;
    int itr;
    for(itr = 0; itr < n_iter; itr++) {
        struct timeval start, mcpy, gemm1, gemm2, ista, diff, logging;
        gettimeofday(&start, NULL);
        

        // for(int idx = 0; idx < dict_sz * n_samples; idx++) {
        //     printf("\tY pre (%d): %f\n", idx, Y[idx]);
        // }


        if(SPARSITY_DEBUG) {
            x_nz = n_nonzero_elements(residual, inp_dim * n_samples);
            basis_nz = n_nonzero_elements(basis, inp_dim * dict_sz);
            y_nz_pre = n_nonzero_elements(Y, n_samples * dict_sz);
        }

        // residual = x - (z @ basis.T)
        memcpy(residual, X, inp_dim * n_samples * sizeof(float));    
        gettimeofday(&mcpy, NULL);
        // Y becomes highly sparse with more iterations. Switch to sparse matmul after initial iterations
        // cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, inp_dim, n_samples, dict_sz, -1.0f, basis, dict_sz, Y, n_samples, 1.0f, residual, n_samples);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n_samples, inp_dim, dict_sz, -1.0f, Y, dict_sz, basis, dict_sz, 1.0f, residual, inp_dim);

        if(SPARSITY_DEBUG)
            res_nz = n_nonzero_elements(residual, inp_dim * n_samples);

        gettimeofday(&gemm1, NULL);

        // mm = residual @ basis
        // z += L_inv * mm
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n_samples, dict_sz, inp_dim, L_inv, residual, inp_dim, basis, dict_sz, 1.0f, Y, dict_sz);


        gettimeofday(&gemm2, NULL);


        // for(int idx = 0; idx < dict_sz * n_samples; idx++) {
        //     printf("\tY (%d): %f\n", idx, Y[idx]);
        // }
        // for(int idx = 0; idx < inp_dim * n_samples; idx++) {
        //     printf("\tres (%d): %f\n", idx, residual[idx]);
        // }


        if(SPARSITY_DEBUG) {
            float y_nz = n_nonzero_elements(Y, n_samples * dict_sz);
            printf("\nsparsity: %.2f %.2f %.2f %.2f %.2f\n", x_nz, basis_nz, y_nz_pre, res_nz, y_nz);
        }

        gettimeofday(&ista, NULL);
        if(DEBUG) {
            log_time_diff("\n\tmcpy", &start, &mcpy);
            log_time_diff("\tgemm1", &mcpy, &gemm1);
            log_time_diff("\tgemm2", &gemm1, &gemm2);
        }
        gettimeofday(&logging, NULL);



        // Y-update multiplier
        tk_prev = tk;
        tk = (1 + sqrtf(1 + 4 * tk * tk)) / 2;
        float mlt = (tk_prev - 1) / tk;

        float diff_norm = 0;
        float prev_z_norm = 0;
        for(int row_idx = 0; row_idx < n_samples; row_idx++) {
            for(int col_idx = 0; col_idx < dict_sz; col_idx++) {
                int idx = row_idx * dict_sz + col_idx;
                
                // apply thresholding
                Y[idx] = Y[idx] < alpha_L ? 0 : Y[idx] - alpha_L;

                float diff = Y[idx] - z_prev[idx];
                diff_norm += diff * diff;
                
                prev_z_norm += z_prev[idx] * z_prev[idx];
                z_prev[idx] = Y[idx];

                Y[idx] += mlt * diff;
            }
        }

        // Frobenius norm can be defined as the L2 norm of the flattened matrix
        float norm_ratio = diff_norm / prev_z_norm;
        norm_ratio = sqrtf(norm_ratio);         // equivalent to sqrtf(diff_norm) / sqrtf(prev_z_norm)


        // printf("%d: %f %f\n", itr, sqrtf(diff_norm), sqrtf(prev_z_norm));




        gettimeofday(&diff, NULL);

        if(DEBUG) {
            log_time_diff("\ntotal", &start, &diff);
            log_time_diff("\tista", &start, &ista);
            log_time_diff("\tlogging", &ista, &logging);
            log_time_diff("\tdiff", &logging, &diff);
        
            printf("\33[2K\r%d / %d", itr, n_iter);
            fflush(stdout);
        }

        if(itr != 0 && norm_ratio < converge_thresh)
            break;
    }

    memcpy(Z, z_prev, dict_sz * n_samples * sizeof(float));

    if(DEBUG)
        printf("\n");

    free(residual);
    free(z_prev);
    free(Y);
    return itr;
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