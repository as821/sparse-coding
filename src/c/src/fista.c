
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

void ista_step(float* X, float* basis, float* Z, float* residual, int inp_dim, int n_samples, int dict_sz, float L_inv, float alpha_L) {
    // residual = x - (basis @ z)
    memcpy(residual, X, inp_dim * n_samples * sizeof(float));    
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, inp_dim, n_samples, dict_sz, -1.0f, basis, dict_sz, Z, n_samples, 1.0f, residual, n_samples);

    // mm = basis.T @ residual
    // z += L_inv * mm
    // NOTE: does not explicitly transpose basis, lets cblas_sgemm to handle it
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, dict_sz, n_samples, inp_dim, L_inv, basis, dict_sz, residual, n_samples, 1.0f, Z, n_samples);

    // z -= mult
    // z = torch.clamp(z, min=0)
    for(int idx = 0; idx < dict_sz * n_samples; idx++) {
        Z[idx] -= alpha_L;
        
        // compiler should be able to make this fast
        // https://stackoverflow.com/questions/427477/fastest-way-to-clamp-a-real-fixed-floating-point-value
        Z[idx] = Z[idx] < 0 ? 0 : Z[idx];
    }
}

void fista(float* X, float* basis, float* Z, int inp_dim, int n_samples, int dict_sz, float L_inv, float alpha_L, int n_iter, float converge_thresh) {
    CHECK(X);
    CHECK(basis);
    CHECK(Z);

    // X: inp_dim x n_samples
    // basis: inp_dim x dict_sz
    // Z: dict_sz x n_samples

    float* residual = (float*) malloc(inp_dim * n_samples * sizeof(float));
    CHECK(residual);

    float* z_prev = (float*) malloc(dict_sz * n_samples * sizeof(float));
    CHECK(z_prev);

    float* Y = (float*) malloc(dict_sz * n_samples * sizeof(float));
    CHECK(Y);
    memcpy(Y, Z, dict_sz * n_samples * sizeof(float));

    float* z_diff = (float*) malloc(dict_sz * n_samples * sizeof(float));
    CHECK(z_diff);


    float tk = 1, tk_prev = 1;
    for(int itr = 0; itr < n_iter; itr++) {

        struct timeval start, ista, y_update, conv;
        gettimeofday(&start, NULL);

        memcpy(z_prev, Z, dict_sz * n_samples * sizeof(float));
        
        // TODO(as) pass Y in rather than Z
        ista_step(X, basis, Y, residual, inp_dim, n_samples, dict_sz, L_inv, alpha_L);
        memcpy(Z, Y, dict_sz * n_samples * sizeof(float));          // TODO(as) can probably skip this except for final iter?
        gettimeofday(&ista, NULL);


        // tk_prev = tk
        // tk = (1 + math.sqrt(1 + 4 * tk ** 2)) / 2
        tk_prev = tk;
        tk = (1 + sqrtf(1 + 4 * powf(tk, 2))) / 2;

        // z_diff = z_slc - prev_z
        // y_slc = z_slc + ((tk_prev - 1) / tk) * z_diff
        float tk_mult = (tk_prev - 1) / tk;
        for(int idx = 0; idx < dict_sz * n_samples; idx++) {
            z_diff[idx] = Y[idx] - z_prev[idx];
            Y[idx] += tk_mult * z_diff[idx];
        }

        gettimeofday(&y_update, NULL);

        // torch.norm(z_diff) / torch.norm(prev_z) < converge_thresh
        if(itr != 0) {
            // Frobenius norm of matrix can be defined as L2 norm of flatttened matrix
            float diff_norm = cblas_snrm2(dict_sz * n_samples, z_diff, 1);
            float prev_norm = cblas_snrm2(dict_sz * n_samples, z_prev, 1);
            if(diff_norm / prev_norm < converge_thresh)
                break;
        }

        gettimeofday(&conv, NULL);

        log_time_diff("\ntotal", &start, &conv);
        log_time_diff("\tista", &start, &ista);
        log_time_diff("\tupdate", &ista, &y_update);
        log_time_diff("\tconv", &y_update, &conv);


        printf("\33[2K\r%d / %d", itr, n_iter);
        fflush(stdout);
    }
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