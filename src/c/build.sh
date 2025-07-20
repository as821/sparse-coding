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

OPTIM_OPTIONS="-O3 -fno-math-errno -fno-trapping-math -march=native"
GCC_OPTIONS="-c -g -I./src/c/src $OPTIM_OPTIONS -o"

gcc -fPIC -fopenmp -I"$OPENBLAS_INCLUDE" $GCC_OPTIONS src/c/bin/fista.o src/c/src/fista.c
gcc -shared -g -L"$OPENBLAS_LIB" $OPTIM_OPTIONS -o src/c/bin/fista.so src/c/bin/fista.o -lopenblas



ADA_OPTIONS="-O3 -arch=sm_89 -gencode=arch=compute_89,code=sm_89 -gencode=arch=compute_89,code=compute_89"
nvcc -shared -g -lcublas -lcuda -Xptxas="-v" -Xptxas="-dlcm=cg" -lineinfo -std=c++17 $ADA_OPTIONS -o src/c/bin/cu_fista_89.so src/c/src/fista.cu -Xcompiler="-fPIC $OPTIM_OPTIONS" -diag-suppress 2464

AMPERE_OPTIONS="-O3 -arch=sm_86 -gencode=arch=compute_86,code=sm_86 -gencode=arch=compute_86,code=compute_86"
nvcc -shared -g -lcublas -lcuda -Xptxas="-v" -Xptxas="-dlcm=cg" -lineinfo -std=c++17 $AMPERE_OPTIONS -o src/c/bin/cu_fista_86.so src/c/src/fista.cu -Xcompiler="-fPIC $OPTIM_OPTIONS" -diag-suppress 2464

# TODO: all the examples just use CMake 
TT_INCLUDE=${TT_METAL_HOME} ${TT_METAL_HOME}/tt_metal ${TT_METAL_HOME}/tt_metal/api/ ${TT_METAL_HOME}/tt_metal/include ${TT_METAL_HOME}/tt_metal/hostdevcommon/api ${TT_METAL_HOME}/tt_metal/third_party/umd ${TT_METAL_HOME}/tt_metal/third_party/umd/device ${TT_METAL_HOME}/tt_metal/third_party/umd/device/api/ ${TT_METAL_HOME}/tt_metal/third_party/tracy/public/ ${TT_METAL_HOME}/tt_metal/hw/inc/ ${TT_METAL_HOME}/tt_stl/"
gcc -fPIC -I"$TT_INCLUDE" $GCC_OPTIONS src/c/bin/tt_fista.o src/c/src/tt/fista.c
gcc -shared -g -L"$TT_METAL_HOME/build/tt_metal" $OPTIM_OPTIONS -o src/c/bin/tt_fista.so src/c/bin/tt_fista.o

rm -f src/c/bin/*.o
rm -rf src/c/bin/*.dSYM