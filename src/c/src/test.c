
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <execinfo.h>

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




void ista_step(float* X, float* basis, float* Z, float* residual, int inp_dim, int n_samples, int dict_sz, float L_inv, float alpha_L) {
    // residual = x - (basis @ z)
    memcpy(residual, X, inp_dim * n_samples * sizeof(float));
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, inp_dim, n_samples, dict_sz, -1.0f, basis, dict_sz, Z, n_samples, 1.0f, residual, n_samples);

    // mm = basis.T @ residual
    // z += L_inv * mm
    // NOTE: does not explicitly transpose basis, lets cblas_sgemm to handle it
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, dict_sz, n_samples, inp_dim, L_inv, basis, dict_sz, residual, n_samples, 1.0f, Z, n_samples);
    free(residual);

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

    ista_step(X, basis, Z, residual, inp_dim, n_samples, dict_sz, L_inv, alpha_L);






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