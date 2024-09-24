import numpy as np
import matplotlib
matplotlib.use('Agg')       # non-interactive backend for matplotlib
import matplotlib.pyplot as plt

from einops import rearrange

def plot_color(rf, out_dim, M):
    normalized = np.zeros_like(rf)
    for i in range(3):  # For each color channel
        channel = rf[:, :,:,i]
        normalized[:, :,:,i] = (channel - channel.min()) / (channel.max() - channel.min())

    n = 10 #int(np.ceil(np.sqrt(normalized.shape[0])))
    fig, axes = plt.subplots(nrows=n, ncols=n, sharex=True, sharey=True)
    fig.set_size_inches(10, 10)
    for i in range(min(n * n, normalized.shape[0])):
        ax = axes[i // n][i % n]
        ax.imshow(normalized[i]) #, cmap='gray', vmin=-1, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')
    if n * n > normalized.shape[0]:
        for j in range(normalized.shape[0], n * n):
            ax = axes[j // n][j % n]
            ax.imshow(np.ones_like(normalized[0]) * -1, cmap='gray', vmin=-1, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect('equal')
    fig.subplots_adjust(wspace=0.0, hspace=0.0)
    return fig

def plot_patch_recon(args, basis, x, z):
    """Visualize the reconstruction of patches from their dictionary elemements"""
    k = 40
    ncol = 4
    fig, axes = plt.subplots(int(k / 2), ncol + 1, figsize=(10, 12))
    fig.subplots_adjust(hspace=0.5, wspace=0.05)
    for i, ax in enumerate(axes.flat):
        ax.axis('off')
    
    patch_sz = z.sum(axis=1)
    arr = np.argsort(patch_sz)

    # don't visualize zero codes
    idx = 0
    while patch_sz[arr[idx]] == 0:
        idx += 1
    arr = arr[idx:]

    # visualize K patches across spectrum of # of activated dictionary elements (including most + least activated)
    running = 0
    for i in range(k):
        # subplot + text formatting
        idx = int((i / k) * arr.shape[0])
        base = i * 2 + running
        if (base % (ncol + 1)) == 2:
            running += 1
            base += 1       # skip middle column
            
            # text on the right side
            fig.text(1-0.075, 0.925 - 0.045*int(i / 2), f'({idx}) {arr[idx]} : {int(patch_sz[arr[idx]])}', ha='center', va='center', fontsize=8)
        else:
            # text on left side
            fig.text(0.075, 0.925 - 0.045*int(i / 2), f'({idx}) {arr[idx]} : {int(patch_sz[arr[idx]])}', ha='center', va='center', fontsize=8)

        # recon
        recon = basis @ z[arr[idx]].T
        recon /= z[arr[idx]].sum()      # average of the dictionary elements
        axes.flat[base].imshow(recon)

        # original patch
        patch = rearrange(x[arr[idx]], "(a b c) -> a b c", a=args.patch_sz, b=args.patch_sz, c=3)
        axes.flat[base + 1].imshow(patch)

    plt.tight_layout(rect=[0.1, 0, 0.9, 0.95])  # left, bottom, right, top
    return fig