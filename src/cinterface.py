import ctypes
import os
import torch
from time import time


def c_impl_available():
    return os.path.exists("src/c/bin/fista.so") and False

def fista(x, basis, alpha, n_iter, converge_thresh=0.01, lr=0.01):
    assert os.path.exists("src/c/bin/fista.so")
    lib = ctypes.CDLL("src/c/bin/fista.so")
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
    lib.fista.restype = None

    z = torch.zeros((x.shape[0], basis.shape[1]), dtype=torch.float32)
    assert x.dtype == torch.float32 and basis.dtype == torch.float32
    assert x.is_contiguous(memory_format=torch.contiguous_format) and basis.is_contiguous(memory_format=torch.contiguous_format) and z.is_contiguous(memory_format=torch.contiguous_format)
    
    start = time()
    lib.fista(ctypes.cast(x.data_ptr(), ctypes.POINTER(ctypes.c_float)), \
                ctypes.cast(basis.data_ptr(), ctypes.POINTER(ctypes.c_float)), \
                ctypes.cast(z.data_ptr(), ctypes.POINTER(ctypes.c_float)), \
                x.shape[0], x.shape[1], basis.shape[1], lr, alpha, n_iter, converge_thresh)
    end = time()

    print(f"FISTA: {end - start:.3f}s")
    return z

