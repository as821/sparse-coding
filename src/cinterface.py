import ctypes
import os
import torch
from time import time


def _get_fista_path():
    if not torch.cuda.is_available():
        return "src/c/bin/fista.so"

    prop = torch.cuda.get_device_properties(0)
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
    assert os.path.exists(_get_fista_path())
    lib = ctypes.CDLL(_get_fista_path())
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


def cu_fista(x, basis, alpha, n_iter, converge_thresh=0.01, lr=0.01, path=_get_fista_path()):
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
        ]
    lib.fista.restype = ctypes.c_int

    z = torch.zeros((x.shape[0], basis.shape[1]), dtype=torch.float32)
    assert x.dtype == torch.float32 and basis.dtype == torch.float32
    assert x.is_contiguous(memory_format=torch.contiguous_format) and basis.is_contiguous(memory_format=torch.contiguous_format) and z.is_contiguous(memory_format=torch.contiguous_format)
    
    start = time()
    n_iter = lib.fista(ctypes.cast(x.data_ptr(), ctypes.POINTER(ctypes.c_float)), \
                ctypes.cast(basis.data_ptr(), ctypes.POINTER(ctypes.c_float)), \
                ctypes.cast(z.data_ptr(), ctypes.POINTER(ctypes.c_float)), \
                x.shape[0], x.shape[1], basis.shape[1], lr, alpha, n_iter, converge_thresh)
    end = time()

    # print(f"FISTA: {end - start:.3f}s")
    return z, n_iter, end - start
