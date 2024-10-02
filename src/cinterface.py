import ctypes
import os
import torch
import threading
import math
from time import time


def _get_fista_path(idx=0, use_cpu=False):
    if not torch.cuda.is_available() or use_cpu:
        return "src/c/bin/fista.so"

    prop = torch.cuda.get_device_properties(idx)
    assert prop.major == 8, "Only Ada and Ampere GPUs are supported."
    if prop.minor == 9:
        return "src/c/bin/cu_fista_89.so"
    elif prop.minor == 6:
        return "src/c/bin/cu_fista_86.so"
    else:
        raise NotImplementedError


def c_impl_available():
    return os.path.exists(_get_fista_path())

def fista(x, basis, alpha, n_iter, converge_thresh=0.01, lr=0.01):
    assert os.path.exists(_get_fista_path(use_cpu=True))
    lib = ctypes.CDLL(_get_fista_path(use_cpu=True))
    lib.fista.argtypes = [
        ctypes.POINTER(ctypes.c_float), 
        ctypes.POINTER(ctypes.c_float), 
        ctypes.POINTER(ctypes.c_float), 
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_int,
        ctypes.c_float,
        ]
    lib.fista.restype = ctypes.c_int

    z = torch.zeros((x.shape[0], basis.shape[1]), dtype=torch.float32)
    assert x.dtype == torch.float32 and basis.dtype == torch.float32
    assert x.is_contiguous(memory_format=torch.contiguous_format) and basis.is_contiguous(memory_format=torch.contiguous_format) and z.is_contiguous(memory_format=torch.contiguous_format)
    
    start = time()
    n_iter, lib.fista(ctypes.cast(x.data_ptr(), ctypes.POINTER(ctypes.c_float)), \
                ctypes.cast(basis.data_ptr(), ctypes.POINTER(ctypes.c_float)), \
                ctypes.cast(z.data_ptr(), ctypes.POINTER(ctypes.c_float)), \
                x.shape[0], x.shape[1], basis.shape[1], lr, alpha, n_iter, converge_thresh)
    end = time()

    print(f"FISTA: {end - start:.3f}s")
    return z, n_iter, end - start

def _get_gpu_ranges(total_work):
    # not all GPUs are equal! Heuristic load balancing (better to do this dynamically)
    n_gpu = torch.cuda.device_count()
    ranges = []
    weight = 0
    for idx in range(n_gpu):
        prop = torch.cuda.get_device_properties(idx)
        assert prop.major == 8, "Only Ada and Ampere GPUs are supported."
        if prop.minor == 9:     # 4090 can do ~2x more work than 3090
            ranges.append(2)
            weight += 2
        elif prop.minor == 6:
            ranges.append(1)
            weight += 1

    prev = 0
    for idx in range(n_gpu):
        amt = math.ceil((ranges[idx] / weight) * total_work)
        ranges[idx] = (prev, prev + amt)
        prev += amt
    assert prev >= total_work
    return ranges

def _thread_func(x, basis, alpha, n_iter, converge_thresh, lr, idx, z, start, end, thr_metadata):
    path = _get_fista_path(idx)
    z_chunk, n_iter, dur = cu_fista(x, basis, alpha, n_iter, converge_thresh, lr, path, idx)
    z[start:end] = z_chunk
    thr_metadata[idx] = (n_iter, dur)


def multi_gpu_cu_fista(x, basis, alpha, n_iter, converge_thresh=0.01, lr=0.01):
    start_time = time()
    n_gpu = torch.cuda.device_count()
    assert n_gpu > 0
    
    if n_gpu > 1:
        ranges = _get_gpu_ranges(x.shape[0])

        z = torch.zeros((x.shape[0], basis.shape[1]), dtype=torch.float32)
        thr_metadata = [None] * n_gpu
        threads = []
        for idx in range(n_gpu):
            start, end = ranges[idx]
            x_chunk = x[start : end]
            threads.append(threading.Thread(target=_thread_func, args=(x_chunk, basis, alpha, n_iter, converge_thresh, lr, idx, z, start, end, thr_metadata)))
            threads[-1].start()

        n_iter = 0
        for idx in range(len(threads)):
            threads[idx].join()
            n_iter = max(n_iter, thr_metadata[idx][0])

        end_time = time()
        return z, n_iter, end_time - start_time
    else:
        return cu_fista(x, basis, alpha, n_iter, converge_thresh, lr, _get_fista_path(), 0)

def cu_fista(x, basis, alpha, n_iter, converge_thresh=0.01, lr=0.01, path=_get_fista_path(), gpu_idx=0):
    assert os.path.exists(path), f"{path}"
    lib = ctypes.CDLL(path)
    lib.fista.argtypes = [
        ctypes.POINTER(ctypes.c_float), 
        ctypes.POINTER(ctypes.c_float), 
        ctypes.POINTER(ctypes.c_float), 
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_int,
        ctypes.c_float,
        ctypes.c_int
        ]
    lib.fista.restype = ctypes.c_int

    z = torch.zeros((x.shape[0], basis.shape[1]), dtype=torch.float32)
    assert x.dtype == torch.float32 and basis.dtype == torch.float32
    assert x.is_contiguous(memory_format=torch.contiguous_format) and basis.is_contiguous(memory_format=torch.contiguous_format) and z.is_contiguous(memory_format=torch.contiguous_format)
    
    start = time()
    n_iter = lib.fista(ctypes.cast(x.data_ptr(), ctypes.POINTER(ctypes.c_float)), \
                ctypes.cast(basis.data_ptr(), ctypes.POINTER(ctypes.c_float)), \
                ctypes.cast(z.data_ptr(), ctypes.POINTER(ctypes.c_float)), \
                x.shape[0], x.shape[1], basis.shape[1], lr, alpha, n_iter, converge_thresh, gpu_idx)
    end = time()

    # print(f"FISTA: {end - start:.3f}s")
    return z, n_iter, end - start
