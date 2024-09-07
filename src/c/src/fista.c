
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

void fista(float* __restrict__ X, float* __restrict__ basis, float* __restrict__ Z, int inp_dim, int n_samples, int dict_sz, float L_inv, float alpha_L, int n_iter, float converge_thresh) {
    CHECK(X);
    CHECK(basis);
    CHECK(Z);

    // X: inp_dim x n_samples
    // basis: inp_dim x dict_sz
    // Z: dict_sz x n_samples

    float* residual = (float*) aligned_alloc(ALIGNMENT, inp_dim * n_samples * sizeof(float));
    CHECK(residual);

    // allow this to contain random values initially since not used on the first iteration
    float* z_prev = (float*) aligned_alloc(ALIGNMENT, dict_sz * n_samples * sizeof(float));
    memset(z_prev, 0, dict_sz * n_samples * sizeof(float));
    CHECK(z_prev);

    float* Y = (float*) aligned_alloc(ALIGNMENT, dict_sz * n_samples * sizeof(float));
    CHECK(Y);
    memset(Y, 0, dict_sz * n_samples * sizeof(float));

    // assumed by vectorization and sparse Y storage
    CHECK(n_samples % 8 == 0);

    float prev_z_norm = 0;
    float tk = 1, tk_prev = 1;
    float x_nz, basis_nz, y_nz_pre, res_nz;
    for(int itr = 0; itr < n_iter; itr++) {
        struct timeval start, mcpy, gemm1, gemm2, ista, diff;
        gettimeofday(&start, NULL);
        
        if(SPARSITY_DEBUG) {
            x_nz = n_nonzero_elements(residual, inp_dim * n_samples);
            basis_nz = n_nonzero_elements(basis, inp_dim * dict_sz);
            y_nz_pre = n_nonzero_elements(Y, n_samples * dict_sz);
        }

        // residual = x - (basis @ z)
        memcpy(residual, X, inp_dim * n_samples * sizeof(float));    
        gettimeofday(&mcpy, NULL);
        // Y becomes highly sparse with more iterations. Switch to sparse matmul after initial iterations
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, inp_dim, n_samples, dict_sz, -1.0f, basis, dict_sz, Y, n_samples, 1.0f, residual, n_samples);

        if(SPARSITY_DEBUG)
            res_nz = n_nonzero_elements(residual, inp_dim * n_samples);

        gettimeofday(&gemm1, NULL);

        // mm = basis.T @ residual
        // z += L_inv * mm
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, dict_sz, n_samples, inp_dim, L_inv, basis, dict_sz, residual, n_samples, 1.0f, Y, n_samples);

        gettimeofday(&gemm2, NULL);


        if(SPARSITY_DEBUG) {
            float y_nz = n_nonzero_elements(Y, n_samples * dict_sz);
            printf("\nsparsity: %.2f %.2f %.2f %.2f %.2f\n", x_nz, basis_nz, y_nz_pre, res_nz, y_nz);
        }

        if(DEBUG) {
            log_time_diff("\n\tmcpy", &start, &mcpy);
            log_time_diff("\tgemm1", &mcpy, &gemm1);
            log_time_diff("\tgemm2", &gemm1, &gemm2);
        }

        gettimeofday(&ista, NULL);

        // thresholding
        __m256 zero_vec = _mm256_set1_ps(0);
        __m256 alpha_L_vec = _mm256_set1_ps(alpha_L);

        // Y-update multiplier
        tk_prev = tk;
        tk = (1 + sqrtf(1 + 4 * tk * tk)) / 2;
        float mlt = (tk_prev - 1) / tk;
        __m256 tk_mult = _mm256_set1_ps(mlt);

        // calculate difference in Z due to this step, calculate norm of the difference and prev. Z, update z_prev 
        __m256 avx_diff_norm = _mm256_set1_ps(0);
        __m256 avx_prev_z_norm = _mm256_set1_ps(0);
        for(int row_idx = 0; row_idx < dict_sz; row_idx++) {             // TODO(as) make sure static scheduling == assigning threads a contiguous block...
            int row_offset = row_idx * n_samples;
            for(int col_idx = 0; col_idx < n_samples; col_idx += 8) {
                int idx = row_offset + col_idx;

                // apply thresholding to the BLAS output
                // Y[idx] = Y[idx] < alpha_L ? 0 : Y[idx] - alpha_L;
                __m256 blas_res_1 = _mm256_load_ps(Y + idx);
                __m256 sub_vec = _mm256_sub_ps(blas_res_1, alpha_L_vec);
                __m256 mask_1 = _mm256_cmp_ps(zero_vec, sub_vec, _CMP_LT_OS);       // A < B (ordered, signalling)
                __m256 Y_val1 = _mm256_blendv_ps(zero_vec, sub_vec, mask_1);        // if mask, then B

                // float diff = Y[idx] - z_prev[idx];
                __m256 diff1 = _mm256_sub_ps(Y_val1, _mm256_load_ps(z_prev + idx));
                
                // y_slc = z_slc + ((tk_prev - 1) / tk) * z_diff
                __m256 Y_next = _mm256_fmadd_ps(tk_mult, diff1, Y_val1);

                // copy Z value out of Y before it gets updated
                // non-temporal store, should bypass cache hierarchy since never accessed again
                _mm256_stream_ps(z_prev + idx, Y_val1);
                _mm256_stream_ps(Y + idx, Y_next);

                // diff_norm += diff * diff;
                avx_diff_norm = _mm256_fmadd_ps(diff1, diff1, avx_diff_norm);        // a * b + c
                
                // Actually the norm of the current Y values, to be used in the next iteration
                // prev_z_norm += z_prev[idx] * z_prev[idx];
                avx_prev_z_norm = _mm256_fmadd_ps(Y_val1, Y_val1, avx_prev_z_norm);
            }
        }

        float diff_norm = horizontal_add(avx_diff_norm);

        // torch.norm(z_diff) / torch.norm(prev_z) < converge_thresh
        // Frobenius norm can be defined as the L2 norm of the flatttened matrix
        float norm_ratio = diff_norm / prev_z_norm;
        norm_ratio = sqrtf(norm_ratio);         // equivalent to sqrtf(diff_norm) / sqrtf(prev_z_norm)

        prev_z_norm = horizontal_add(avx_prev_z_norm);
        gettimeofday(&diff, NULL);

        if(DEBUG) {
            log_time_diff("\ntotal", &start, &diff);
            log_time_diff("\tista", &start, &ista);
            log_time_diff("\tdiff", &ista, &diff);
        
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