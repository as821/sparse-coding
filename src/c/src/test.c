
#include <stdio.h>

void fista(float* x, float* basis, float* z, int n_samples, int dict_sz, int inp_dim, float alpha, int n_iter, float converge_thresh) {
    printf("Recv: %d %d %d %f %d %f\n", n_samples, dict_sz, inp_dim, alpha, n_iter, converge_thresh);
}
