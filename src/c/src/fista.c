
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <execinfo.h>
#include <sys/time.h>

#include <math.h>
#include <cblas.h>

#include <immintrin.h>


void print_stack_trace();

#define ALIGNMENT 32            // bytes --> 256 bits

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

    printf("%s: %f\n", msg, diff_in_sec);
}

void ista_step(float* __restrict__ X, float* __restrict__ basis, float* __restrict__ Z, float* __restrict__ residual, int inp_dim, int n_samples, int dict_sz, float L_inv, float alpha_L) {
    // residual = x - (basis @ z)

    struct timeval start, gemm1, gemm2, loop;
    gettimeofday(&start, NULL);


    memcpy(residual, X, inp_dim * n_samples * sizeof(float));    
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, inp_dim, n_samples, dict_sz, -1.0f, basis, dict_sz, Z, n_samples, 1.0f, residual, n_samples);
    gettimeofday(&gemm1, NULL);


    // mm = basis.T @ residual
    // z += L_inv * mm
    // NOTE: does not explicitly transpose basis, lets cblas_sgemm to handle it
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, dict_sz, n_samples, inp_dim, L_inv, basis, dict_sz, residual, n_samples, 1.0f, Z, n_samples);
    gettimeofday(&gemm2, NULL);

    // z -= mult
    // z = torch.clamp(z, min=0)
    int step = 8;
    CHECK(dict_sz * n_samples % step == 0);        // 256 bit registers -> 32 bytes -> 8 floats
    float* aligned_Z = (float*) __builtin_assume_aligned(Z, ALIGNMENT);
    __m256 zero_vec = _mm256_set1_ps(0);
    __m256 alpha_L_vec = _mm256_set1_ps(alpha_L);
    __m256 neg_alpha_L_vec = _mm256_set1_ps(-1 * alpha_L);
    for(int idx = 0; idx < dict_sz * n_samples; idx += step) {
        // aligned_Z[idx] = aligned_Z[idx] < alpha_L ? 0 : aligned_Z[idx] - alpha_L;
        float* ptr = aligned_Z + idx;
        __m256 cur_val = _mm256_load_ps(ptr);
        __m256 false_branch_val = _mm256_add_ps(cur_val, neg_alpha_L_vec);
        __m256 mask = _mm256_cmp_ps(cur_val, alpha_L_vec, _CMP_LE_OS);          // A <= B (ordered, signalling)
        __m256 result = _mm256_blendv_ps(false_branch_val, zero_vec, mask);      // if mask, then B
        _mm256_store_ps(ptr, result);
    }

    gettimeofday(&loop, NULL);

    log_time_diff("\n\tgemm1", &start, &gemm1);
    log_time_diff("\tgemm2", &gemm1, &gemm2);
    log_time_diff("\tloop", &gemm2, &loop);
}


float horizontal_add(__m256 v) {
    // add the high and low 128 bits
    __m128 sum128 = _mm_add_ps(_mm256_extractf128_ps(v, 1), _mm256_castps256_ps128(v));
    
    // horizontal add the 4 floats in the 128-bit vector
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    return _mm_cvtss_f32(sum128);
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
    CHECK(z_prev);

    float* Y = (float*) aligned_alloc(ALIGNMENT, dict_sz * n_samples * sizeof(float));
    CHECK(Y);
    memcpy(Y, Z, dict_sz * n_samples * sizeof(float));

    float* z_diff = (float*) aligned_alloc(ALIGNMENT, dict_sz * n_samples * sizeof(float));
    CHECK(z_diff);

    // assumed by vectorization inside this loop
    CHECK(dict_sz * n_samples % 8 == 0);

    float tk = 1, tk_prev = 1;
    for(int itr = 0; itr < n_iter; itr++) {
        struct timeval start, ista, y_update, diff;
        gettimeofday(&start, NULL);
        
        ista_step(X, basis, Y, residual, inp_dim, n_samples, dict_sz, L_inv, alpha_L);
        gettimeofday(&ista, NULL);

        // calculate difference in Z due to this step, calculate norm of the difference and prev. Z, update z_prev 
        __m256 avx_diff_norm = _mm256_set1_ps(0);
        __m256 avx_prev_z_norm = _mm256_set1_ps(0);

        #pragma omp parallel
        {
            // __m256 avx_diff_norm_local = _mm256_set1_ps(0);
            // __m256 avx_prev_z_norm_local = _mm256_set1_ps(0);

            struct {
                __m256 avx_diff_norm_local;
                __m256 avx_prev_z_norm_local;
                char padding[32];  // Ensure 64-byte alignment
            } local_data __attribute__((aligned(64))) = {_mm256_set1_ps(0), _mm256_set1_ps(0)};         // TODO(as) check cache line size...

            
            #pragma omp for nowait
            for(int idx = 0; idx < dict_sz * n_samples; idx += 16) {
                // float diff = Y[idx] - z_prev[idx];
                __m256 Y_val1 = _mm256_load_ps(Y + idx);
                __m256 Y_val2 = _mm256_load_ps(Y + idx + 8);
                __m256 z_prev_val1 = _mm256_load_ps(z_prev + idx);
                __m256 z_prev_val2 = _mm256_load_ps(z_prev + idx + 8);
                
                __m256 diff1 = _mm256_sub_ps(Y_val1, z_prev_val1);
                __m256 diff2 = _mm256_sub_ps(Y_val2, z_prev_val2);
                
                // diff_norm += diff * diff;
                local_data.avx_diff_norm_local = _mm256_fmadd_ps(diff1, diff1, local_data.avx_diff_norm_local);       // a * b + c
                local_data.avx_diff_norm_local = _mm256_fmadd_ps(diff2, diff2, local_data.avx_diff_norm_local);
                
                // prev_z_norm += z_prev[idx] * z_prev[idx];
                local_data.avx_prev_z_norm_local = _mm256_fmadd_ps(z_prev_val1, z_prev_val1, local_data.avx_prev_z_norm_local);
                local_data.avx_prev_z_norm_local = _mm256_fmadd_ps(z_prev_val2, z_prev_val2, local_data.avx_prev_z_norm_local);
                
                // copy Z value out of Y before it gets updated
                // z_prev[idx] = Y[idx];
                // z_diff[idx] = diff;
                _mm256_stream_ps(z_prev + idx, Y_val1);         // non-temporal store, should bypass cache hierarchy since never accessed again
                _mm256_stream_ps(z_prev + idx + 8, Y_val2);     // TODO(as) can we move this into the update loop instead?
                _mm256_stream_ps(z_diff + idx, diff1);
                _mm256_stream_ps(z_diff + idx + 8, diff2);
            }

            #pragma omp critical
            {
                avx_diff_norm = _mm256_add_ps(avx_diff_norm, local_data.avx_diff_norm_local);
                avx_prev_z_norm = _mm256_add_ps(avx_prev_z_norm, local_data.avx_prev_z_norm_local);                
            }
        }

        float diff_norm = horizontal_add(avx_diff_norm);
        float prev_z_norm = horizontal_add(avx_prev_z_norm);

        // torch.norm(z_diff) / torch.norm(prev_z) < converge_thresh
        // Frobenius norm can be defined as the L2 norm of the flatttened matrix
        float norm_ratio = diff_norm / prev_z_norm;
        norm_ratio = sqrtf(norm_ratio);         // equivalent to sqrtf(diff_norm) / sqrtf(prev_z_norm)
        if(itr != 0 && norm_ratio < converge_thresh)
            break;

        gettimeofday(&diff, NULL);

        // perform Y update only if another ISTA iter needs to be performed. otherwise, Y will contain the final results that should be copied to the output matrix
        if(itr != n_iter - 1) {
            // tk_prev = tk
            // tk = (1 + math.sqrt(1 + 4 * tk ** 2)) / 2
            tk_prev = tk;
            tk = (1 + sqrtf(1 + 4 * tk * tk)) / 2;

            // y_slc = z_slc + ((tk_prev - 1) / tk) * z_diff
            float tk_mult = (tk_prev - 1) / tk;
            for(int idx = 0; idx < dict_sz * n_samples; idx++) {
                Y[idx] += tk_mult * z_diff[idx];
            }
        }

        gettimeofday(&y_update, NULL);

        log_time_diff("\ntotal", &start, &y_update);
        log_time_diff("\tista", &start, &ista);
        log_time_diff("\tdiff", &ista, &diff);
        log_time_diff("\tupdate", &diff, &y_update);


        printf("\33[2K\r%d / %d", itr, n_iter);
        fflush(stdout);
    }

    memcpy(Z, Y, dict_sz * n_samples * sizeof(float));          // TODO(as) can probably skip this except for final iter?

    printf("\n");

    free(residual);
    free(z_prev);
    free(Y);
    free(z_diff);

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