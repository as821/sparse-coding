if [[ "$(uname)" == "Darwin" ]]; then
    # macOS
    OPENBLAS_INCLUDE="/usr/local/opt/openblas/include"
    OPENBLAS_LIB="/usr/local/opt/openblas/lib"
elif [[ "$(uname)" == "Linux" ]]; then
    # Linux
    OPENBLAS_INCLUDE="/usr/include/x86_64-linux-gnu/openblas-pthread"
    OPENBLAS_LIB="/usr/lib/x86_64-linux-gnu/openblas-pthread/libopenblas.a"
else
    echo "Unsupported operating system"
    exit 1
fi

# Set common options
OPTIM_OPTIONS="-O3 -fno-math-errno -fno-trapping-math -march=native"
GCC_OPTIONS="-c -g -I./src/c/src $OPTIM_OPTIONS -o"

gcc -fPIC -I"$OPENBLAS_INCLUDE" $GCC_OPTIONS src/c/bin/fista.o src/c/src/fista.c
gcc -shared -g -L"$OPENBLAS_LIB" $OPTIM_OPTIONS -o src/c/bin/fista.so src/c/bin/fista.o -lopenblas

rm -f src/c/bin/*.o
rm -rf src/c/bin/*.dSYM