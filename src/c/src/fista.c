
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <execinfo.h>
#include <sys/time.h>

#include <math.h>
#include <cblas.h>


void print_stack_trace();

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
    for(int idx = 0; idx < dict_sz * n_samples; idx++) {        
        // compiler should be able to make this fast
        // https://stackoverflow.com/questions/427477/fastest-way-to-clamp-a-real-fixed-floating-point-value
        Z[idx] = Z[idx] < alpha_L ? 0 : Z[idx] - alpha_L;
    }

    gettimeofday(&loop, NULL);

    log_time_diff("\n\tgemm1", &start, &gemm1);
    log_time_diff("\tgemm2", &gemm1, &gemm2);
    log_time_diff("\tloop", &gemm2, &loop);
}

void fista(float* __restrict__ X, float* __restrict__ basis, float* __restrict__ Z, int inp_dim, int n_samples, int dict_sz, float L_inv, float alpha_L, int n_iter, float converge_thresh) {
    CHECK(X);
    CHECK(basis);
    CHECK(Z);

    // X: inp_dim x n_samples
    // basis: inp_dim x dict_sz
    // Z: dict_sz x n_samples

    size_t alloc_alignment = 32;        // bytes -> 256 bits
    float* residual = (float*) aligned_alloc(alloc_alignment, inp_dim * n_samples * sizeof(float));
    CHECK(residual);

    // allow this to contain random values initially since not used on the first iteration
    float* z_prev = (float*) aligned_alloc(alloc_alignment, dict_sz * n_samples * sizeof(float));
    CHECK(z_prev);

    float* Y = (float*) aligned_alloc(alloc_alignment, dict_sz * n_samples * sizeof(float));
    CHECK(Y);
    memcpy(Y, Z, dict_sz * n_samples * sizeof(float));

    float* z_diff = (float*) aligned_alloc(alloc_alignment, dict_sz * n_samples * sizeof(float));
    CHECK(z_diff);


    float tk = 1, tk_prev = 1;
    for(int itr = 0; itr < n_iter; itr++) {
        struct timeval start, ista, y_update, diff;
        gettimeofday(&start, NULL);
        
        ista_step(X, basis, Y, residual, inp_dim, n_samples, dict_sz, L_inv, alpha_L);
        gettimeofday(&ista, NULL);

        // z_diff = z_slc - prev_z
        float diff_norm = 0;
        float prev_z_norm = 0;
        for(int idx = 0; idx < dict_sz * n_samples; idx++) {
            float diff = Y[idx] - z_prev[idx];

            // norm of z_diff and z_prev
            diff_norm += diff * diff;
            z_diff[idx] = diff;
            prev_z_norm += z_prev[idx] * z_prev[idx];

            // copy Z value out of Y before it gets updated
            z_prev[idx] = Y[idx];
        }
        
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