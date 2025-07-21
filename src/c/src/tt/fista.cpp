
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <execinfo.h>
#include <sys/time.h>
#include <stdbool.h>
#include <math.h>



// #include <fmt/ostream.h>
#include <cstdint>
#include <random>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/bfloat16.hpp>

#include "matmul.h"

using namespace tt::tt_metal;


void print_stack_trace();


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



std::shared_ptr<Buffer> alloc_dram_buffer(IDevice* device, const uint32_t num_el) {
    // assume evenly divisible by tile size
    const uint32_t el_per_tile = tt::constants::TILE_WIDTH * tt::constants::TILE_HEIGHT;
    CHECK(num_el % el_per_tile == 0);
    const uint32_t tile_size_bytes = sizeof(bfloat16) * el_per_tile;
    const uint32_t size_bytes = sizeof(bfloat16) * num_el;

    tt::tt_metal::InterleavedBufferConfig dram_config{
        .device = device,
        .size = size_bytes,
        .page_size = tile_size_bytes,
        .buffer_type = tt::tt_metal::BufferType::DRAM
    };
    return CreateBuffer(dram_config);
}



extern "C" 
{
    int fista(float* __restrict__ X_float, float* __restrict__ basis_float, float* __restrict__ Z_float, int n_samples, int inp_dim, int dict_sz, float lr, float alpha_L, int n_iter, float converge_thresh, int gpu_idx) {
        CHECK(X_float);
        CHECK(basis_float);
        CHECK(Z_float);

        // X: n_samples x inp_dim
        // basis: inp_dim x dict_sz
        // Z: n_samples x dict_sz

        IDevice* device = CreateDevice(gpu_idx);
        CommandQueue& cq = device->command_queue();
        Program program = CreateProgram();

        // naively convert all inputs from float -> BF16 for now
        // TODO: look into higher precision options down the line
        int x_n_el = n_samples * inp_dim;
        int basis_n_el = inp_dim * dict_sz;
        int z_n_el = n_samples * dict_sz;
        bfloat16* X_host = malloc(sizeof(bfloat16) * x_n_el);
        CHECK(X_host);
        bfloat16* basis_host = malloc(sizeof(bfloat16) * basis_n_el);
        CHECK(basis_host);        
        bfloat16* Z_host = malloc(sizeof(bfloat16) * z_n_el);
        CHECK(Z_host);

        // TODO: write a float -> bfloat16 copy kernel



        // allocate device buffers and perform host -> device transfers
        auto X_buf = alloc_dram_buffer(device, x_n_el);
        auto basis_buf = alloc_dram_buffer(device, basis_n_el);
        auto Z_buf = alloc_dram_buffer(device, z_n_el);
        auto residual_buf = alloc_dram_buffer(device, x_n_el);
        EnqueueWriteBuffer(cq, X_buf, X_host, false);
        EnqueueWriteBuffer(cq, basis_buf, basis_host, false);
        EnqueueWriteBuffer(cq, Z_buf, Z_host, false);

        float tk = 1, tk_prev = 1;
        int itr;
        for(itr = 0; itr < n_iter; itr++) {

            // TODO: copy x into residual (device -> device transfer)
            

            // residual = x - (z @ basis.T)
            matmul(Z, basis, residual, -1, false, true);
            
            // mm = residual @ basis
            // z += lr * mm
            matmul(residual, basis, Z, lr, false, false);

            // TODO: y-update
            tk_prev = tk;
            tk = (1 + sqrtf(1 + 4 * tk * tk)) / 2;
            float mlt = (tk_prev - 1) / tk;


            // TODO: norm-based termination check
            // Frobenius norm can be defined as the L2 norm of the flattened matrix
            float diff_norm = norms_host[0];
            float prev_z_norm = norms_host[1];
            float norm_ratio = diff_norm / prev_z_norm;
            norm_ratio = sqrtf(norm_ratio);         // equivalent to sqrtf(diff_norm) / sqrtf(prev_z_norm)

            if(itr != 0 && norm_ratio < converge_thresh)
                break;
        }

        // TODO: clean up

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