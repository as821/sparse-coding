
import torch
from time import time
import math
from tqdm import tqdm

def torch_positive_only(basis, x, z, L_inv, mult):    
    residual = x - (z @ basis.T)
    mm = residual @ basis
    z += L_inv * mm
    z -= mult
    z = torch.clamp(z, min=0)
    return z


def FISTA(x, basis, alpha, num_iter, converge_thresh=0.01, device="cpu", tqdm_disable=False, lr=0.001):
    start_time = time()
    z = torch.zeros((x.shape[0], basis.shape[1]), dtype=basis.dtype, device=device)
    x = x.to(device)

    prev_z = torch.zeros_like(z)
    y_slc = z.clone()

    tk, tk_prev = 1, 1
    for itr in range(0, num_iter):
        z = torch_positive_only(basis, x, y_slc.clone(), lr, alpha)

        tk_prev = tk
        tk = (1 + math.sqrt(1 + 4 * tk ** 2)) / 2

        z_diff = z - prev_z
        y_slc = z + ((tk_prev - 1) / tk) * z_diff

        # print(f"py ({itr}): {torch.norm(z_diff)} {torch.norm(prev_z)}")

        if torch.norm(z_diff) / torch.norm(prev_z) < converge_thresh and itr > 0:
            break
        prev_z = z.clone()

    if not tqdm_disable:
        print(f"FISTA iters: {itr} / {num_iter} in {time() - start_time:.3f}s")
    return z.to(device), itr