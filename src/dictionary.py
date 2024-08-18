import os
import sys
import torch
import numpy as np
from tqdm import tqdm
import random
import matplotlib
matplotlib.use('Agg')       # non-interactive backend for matplotlib
import matplotlib.pyplot as plt
from einops import rearrange
from time import time

import random
import wandb
import math

def converged_quadratic_basis_update(basis, x, z, device, lr, idx, warmup_dur, vis_dict, converge_thresh=1e-5, max_iter=20000):
    # warm up the LR
    if idx < warmup_dur and warmup_dur > 0:
        scale = ((idx+1) / (warmup_dur+1))      # +1 so first step is >0
        lr = scale * lr
        # print(f"LR scale: {scale:4f}. Effective LR: {lr:4f} ({idx} / {warmup_dur})")
    vis_dict["lr"] = lr

    z = z.to(device)
    x = x.to(device)

    prev_d_basis = None
    diff = 0

    xzt = x @ z.T
    zzt = z @ z.T

    for itr in range(max_iter):
        # residual = x - basis @ z
        # d_basis = residual @ z.T

        d_basis = xzt - basis @ zzt

        basis += lr * d_basis
        basis /= torch.linalg.vector_norm(basis, dim=0)
        
        if itr % 50 == 0:
            if prev_d_basis is not None:
                diff = (prev_d_basis - d_basis).abs().max()
                prev_d_basis
            prev_d_basis = d_basis.clone()
                
            if diff < converge_thresh and itr > 0:
                break

        if itr % 1000 == 0:
            print(f"\titr {itr}: {d_basis.abs().mean()}")

    print(f"Basis update terminated: ({itr} == {max_iter}) ({diff} < {converge_thresh}). {d_basis.abs().mean()}")
    return basis, vis_dict

def FISTA(x, basis, device, alpha, num_iter, pos_only=True, converge_thresh=0.001, batch_sz=256, tqdm_disable=False):
    start_time = time()

    # L is upper bound on the largest eigenvalue of the basis matrix
    L = torch.max(torch.linalg.eigvalsh(basis @ basis.T))
    mult = alpha / L
    L_inv = 1./L

    def positive_only(basis, x, z, L_inv, mult):
        residual = x - (basis @ z)
        mm = basis.T @ residual
        z += L_inv * mm
        z -= mult
        z = torch.clamp(z, min=0)
        return z

    def not_positive_only(basis, x, z, L_inv, mult):
        residuals = x - (basis @ z)
        mm = basis.T @ residuals
        z += L_inv * mm
        sign = torch.sign(z)
        z = torch.abs(z)
        z -= mult
        z = torch.clamp(z, min=0)
        z *= sign
        return z

    z = torch.zeros((basis.shape[1], x.shape[1]), dtype=basis.dtype, device="cpu")

    compiled = positive_only
    if torch.cuda.is_available():
        z = z.pin_memory()
        compiled = torch.compile(positive_only, dynamic=False, fullgraph=True, options=\
        {
            'max_autotune': True, 
            'max_autotune_pointwise': True,
            'max_autotune_gemm': True,
            'triton.cudagraphs': False,
            'shape_padding': True,
            'epilogue_fusion': True,
            'search_autotune_cache': False,
            'permute_fusion': True,
        })

    max_itr = -1
    for start in tqdm(range(0, x.shape[1], batch_sz), disable=tqdm_disable):
        end = min(start + batch_sz, x.shape[1])
        
        z_slc = z[:, start:end].to(device, non_blocking=True)
        x_slc = x[:, start:end].to(device, non_blocking=True)
        prev_z = torch.zeros_like(z_slc)
        y_slc = z_slc.clone()

        tk = 1
        tk_prev = 1

        block_sz = 1
        for itr in range(0, num_iter):

            z_slc = compiled(basis, x_slc, y_slc, L_inv, mult)

            tk_prev = tk
            tk = (1 + math.sqrt(1 + 4 * tk ** 2)) / 2

            z_diff = z_slc - prev_z
            y_slc = z_slc + ((tk_prev - 1) / tk) * z_diff

            if itr % block_sz == 0:
                prev_diff = torch.abs(z_diff).max() / block_sz
                if prev_diff < converge_thresh and itr > 0:
                    break
            prev_z = z_slc.clone()


        z[:, start:end] = z_slc.to("cpu", non_blocking=True)

        
        max_itr = max(max_itr, itr)
        # print(f"{itr}: {prev_diff}")
    # if not tqdm_disable:
    print(f"FISTA iters: {max_itr} / {num_iter} in {time() - start_time:.3f}s")
    return z

def generate_sparse_coding_dict(args, x, dict_sz, dset):    
    if args.wandb:
        wandb.init(config=args, project="smt_sc_dict")

    # randomly init basis, L2 normalize elements
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    basis = torch.randn((x.shape[0], dict_sz), device=device)
    basis = basis / basis.norm(2,0)

    history_len = 300
    signal = 0
    noise = 0
    moving_avg_mult = (history_len - 1) / history_len
    signal_step = torch.pow(x,2).sum() / history_len

    if torch.cuda.is_available():
        x = x.pin_memory()

    z, batch = None, None
    for i in range(args.sc_dict_itr):
        vis_dict = {}
        assert args.sc_mini > 0:

        # use SGD rather than full-dataset GD. select random minibatches
        idxs = list(range(0, x.shape[1]))
        random.shuffle(idxs)
        sz = len(range(0, x.shape[1], args.sc_mini))
        for idx, start in enumerate(tqdm(range(0, x.shape[1], args.sc_mini))):
            end = min(start + args.sc_mini, x.shape[1])
            batch = x[:, idxs[start:end]]

            z = FISTA(batch, basis, device, args.sc_lambda, args.sc_ista_itr, converge_thresh=args.sc_ista_tol, batch_sz=args.sc_ista_batch, tqdm_disable=True)
            basis, vis_dict = converged_quadratic_basis_update(basis, batch, z, device, args.sc_lr, i * sz + idx, args.sc_lr_warm, vis_dict)
    
        # calculate residual of last batch for visualization
        residual = batch - basis.cpu() @ z
        signal, noise = logging(args, dset, z, residual, basis, vis_dict, i, signal, noise, history_len, moving_avg_mult, signal_step)
    if args.wandb:
        wandb.finish()
    return basis.cpu(), None

def dict_vis(args, dset, basis, itr, vis_dict, n_img=400):
    if args.dict_sz[0] < n_img:
        n_img = int(np.sqrt(args.dict_sz[0])) ** 2
    nrow = ncol = int(np.ceil(np.sqrt(n_img)))
    chunk_sz = args.patch_sz + 1
    img = np.empty([chunk_sz * nrow - 1, chunk_sz * ncol - 1, 3])
    img.fill(0)

    fig = plt.figure()
    ax = fig.gca()

    for i in range(nrow):
        for j in range(ncol):
            p = basis[:, i * ncol + j].to("cpu")

            # vis
            # p = dset.preproc_to_orig_general(p)
            patch = rearrange(p.squeeze(), "(a b c) -> a b c", a=dset.n_inp_channels, b=args.patch_sz, c=args.patch_sz)
            patch = patch.permute((1, 2, 0))
            patch = patch.numpy()

            img[chunk_sz * i : chunk_sz * (i+1) - 1, chunk_sz * j : chunk_sz * (j+1) - 1, :] = patch
                
    ax.imshow(img)
    ax.get_yaxis().set_visible(False)
    ax.get_xaxis().set_visible(False)
    ax.spines["right"].set_color("none")
    ax.spines["top"].set_color("none")
    ax.spines["left"].set_color("none")
    ax.spines["bottom"].set_color("none")

    if args.wandb:
        vis_dict["raw_dict"] = wandb.Image(plt)
        return vis_dict
    elif args.vis_dir != '':
        plt.savefig(get_vis_path(args, "step_" + str(itr)))
        plt.close()
        return None
    else:
        plt.show()
        return None

def get_vis_path(args, name):
    if args.dict_type == "sc":
        dir = args.vis_dir + f"/d{args.dict_sz[0]}_l{args.sc_lambda}_l{args.sc_lr}_i{args.sc_ista_itr}_it{args.sc_dict_itr}_s{args.samples}_c{args.context_sz[0]}_wt{args.whiten_tol}_dw{int(args.disable_whiten)}"
    else:
        dir = args.vis_dir + f"/d{args.dict_sz[0]}_dt{args.dict_thresh[0]}_g{args.gq_thresh[0]}_s{args.samples}_c{args.context_sz[0]}_wt{args.whiten_tol}_dw{int(args.disable_whiten)}"
    if not os.path.exists(dir):
        os.mkdir(dir)
    return dir + "/" + name

def logging(args, dset, z, residual, basis, vis_dict, i, signal, noise, history_len, moving_avg_mult, signal_step):
    signal = (signal * moving_avg_mult) + signal_step
    noise = (noise * moving_avg_mult) + torch.pow(residual,2).sum() / history_len
    snr = signal / noise

    s = f"Step {i}:"
    s += f"\n\tSNR: {snr.item()}" 
    #\n\tMax Hess. Diag: {hessian_diag.max().item()}"
    nz_z = z[z != 0]
    sz = nz_z.size()[0]
    nz = (z != 0).to(int).sum(dim=0)
    s += f"\n\t# activations: {sz} ({(sz / (z.shape[0]*z.shape[1])):.4f}%)"
    nz_mx = nz.max()
    nz_mn = nz.min()
    nz_mean = nz.float().mean()
    z_mx = z.max()
    z_mn = nz_z.min()
    z_avg = z.mean()
    abs_res = residual.abs()
    res_mx = abs_res.max()
    res_mean = abs_res.mean()
    res_sum = abs_res.sum()
    res_med = residual[residual!=0].abs().median()
    s += f"\n\tmax activ: {nz_mx}"
    s += f"\n\tmin activ: {nz_mn}"
    s += f"\n\tmean activ: {nz_mean}"
    s += f"\n\tL1 max: {z_mx}\n\tL1 min: {z_mn}\n\tL1 avg: {z_avg}"
    s += f"\n\tRes max: {res_mx}\n\tRes mean: {res_mean}\n\tRes med: {res_med}\n\tRes sum: {res_sum}"
    print(s)

    if i % 5 == 0:
        vis_dict = dict_vis(args, dset, basis, i, vis_dict)

    if args.wandb:
        vis_dict["snr"] = snr.item()
        vis_dict["n_activations"] = sz
        vis_dict["perc_activations"] = sz / (z.shape[0]*z.shape[1])
        vis_dict["max_activations"] = nz_mx
        vis_dict["min_activations"] = nz_mn
        vis_dict["avg_activations"] = nz_mean
        vis_dict["max_l1"] = z_mx
        vis_dict["min_l1"] = z_mn
        vis_dict["mean_l1"] = z_avg
        vis_dict["max_res"] = res_mx
        vis_dict["mean_res"] = res_mean
        vis_dict["med_res"] = res_med
        vis_dict["res_sum"] = res_sum
        wandb.log(vis_dict, step=i)
    return signal, noise