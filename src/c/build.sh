if [[ "$(uname)" == "Darwin" ]]; then
    # macOS
    OPENBLAS_INCLUDE="/usr/local/opt/openblas/include"
    OPENBLAS_LIB="/usr/local/opt/openblas/lib"
elif [[ "$(uname)" == "Linux" ]]; then
    # Linux
    OPENBLAS_INCLUDE="/home/astange/OpenBLAS"
    OPENBLAS_LIB="/home/astange/OpenBLAS/libopenblas_zenp-r0.3.28.dev.a"
else
    echo "Unsupported operating system"
    exit 1
fi

# Set common options
OPTIM_OPTIONS="-O3 -fno-math-errno -fno-trapping-math -march=native"
GCC_OPTIONS="-c -g -I./src/c/src $OPTIM_OPTIONS -o"

gcc -fPIC -fopenmp -I"$OPENBLAS_INCLUDE" $GCC_OPTIONS src/c/bin/fista.o src/c/src/fista.c
gcc -shared -g -L"$OPENBLAS_LIB" $OPTIM_OPTIONS -o src/c/bin/fista.so src/c/bin/fista.o -lopenblas

nvcc -shared -g -lcublas -o src/c/bin/cu_fista.so src/c/src/fista.cu -Xcompiler="-fPIC $OPTIM_OPTIONS"

rm -f src/c/bin/*.o
rm -rf src/c/bin/*.dSYM