import numpy as np
import ctypes
import os
from time import time


def fista(x, basis, alpha, n_iter, converge_thresh=0.01):
    assert os.path.exists("src/c/bin/test.so")
    lib = ctypes.CDLL("src/c/bin/test.so")
    lib.fista.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'), 
        np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'), 
        np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'), 
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_int,
        ctypes.c_float,
        ]
    lib.fista.restype = None

    z = np.zeros((basis.shape[1], x.shape[1]), dtype=np.float32)
    assert x.dtype == np.float32 and basis.dtype == np.float32

    # L is upper bound on the largest eigenvalue of the basis matrix
    L = np.max(np.linalg.eigvalsh(basis @ basis.T))
    alpha_L = alpha / L
    L_inv = 1./L

    start = time()
    lib.fista(x, basis, z, x.shape[0], x.shape[1], basis.shape[1], L_inv, alpha_L, n_iter, converge_thresh)
    end = time()

    print(f"FISTA: {end - start:.3f}s")
    return z

