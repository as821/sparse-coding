
import torch
from time import time
import math
from tqdm import tqdm

def torch_positive_only(basis, x, z, L_inv, mult):
    residual = x - (basis @ z)
    mm = basis.T @ residual
    z += L_inv * mm
    z -= mult
    z = torch.clamp(z, min=0)       # vs. F.relu
    return z


def FISTA(x, basis, alpha, num_iter, converge_thresh=0.01, batch_sz=256, tqdm_disable=False):
    start_time = time()

    # L is upper bound on the largest eigenvalue of the basis matrix
    # L = torch.max(torch.linalg.eigvalsh(basis @ basis.T))
    # mult = alpha / L
    # L_inv = 1./L

    L_inv = 0.01
    mult = alpha



    z = torch.zeros((basis.shape[1], x.shape[1]), dtype=basis.dtype, device="cpu")

    max_itr = -1
    for start in tqdm(range(0, x.shape[1], batch_sz), disable=tqdm_disable):
        end = min(start + batch_sz, x.shape[1])
        
        z_slc = z[:, start:end]
        x_slc = x[:, start:end]

        prev_z = torch.zeros_like(z_slc)
        y_slc = z_slc.clone()

        tk, tk_prev = 1, 1
        for itr in range(0, num_iter):
            z_slc = torch_positive_only(basis, x_slc, y_slc.clone(), L_inv, mult)

            tk_prev = tk
            tk = (1 + math.sqrt(1 + 4 * tk ** 2)) / 2

            z_diff = z_slc - prev_z
            y_slc = z_slc + ((tk_prev - 1) / tk) * z_diff

            if torch.norm(z_diff) / torch.norm(prev_z) < converge_thresh and itr > 0:
                break
            prev_z = z_slc.clone()
        z[:, start:end] = z_slc

        max_itr = max(max_itr, itr)
    if not tqdm_disable:
        print(f"FISTA iters: {max_itr} / {num_iter} in {time() - start_time:.3f}s")
    return z