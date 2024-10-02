# Sparse Coding

This is a sparse coding implementation for Zen 3 CPUs and for Ampere/Ada GPUs. The CPU implementation uses OpenBLAS and AVX256 instructions while the CUDA implementation relies on a mix of cuBLAS and custom elementwise kernels.

Fast Iterative Shrinkage Thresholding Algorithm (FISTA) is used in both implementations.

https://www.ceremade.dauphine.fr/~carlier/FISTA



## Usage
Update OpenBLAS library path in build script before building.

```
mkdir src/c/bin
./src/c/build.sh 
```

Small sanity check against baseline Python implementation on original Olshausen and Field (1996) dataset:

https://www.rctn.org/bruno/sparsenet/

``` 
python scripts/test_fista.py --path=<> 
```

Full dictionary learning script, for use with CIFAR10 or natural image dataset:

``` 
python scripts/learn_dict.py --path=data/cifar10 --ckpt-path=ckpts --nsamples=5000000 --patch_sz=6 --epoch=20 --alpha=0.0005 --dict_sz=8192 --batch_sz=16384 --fista_conv=0.0001 
```



