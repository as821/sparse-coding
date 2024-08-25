# compile C files into shared library. call from top-level directory of repository
OPTIM_OPTIONS="-O3 -fno-math-errno -fno-trapping-math -march=native"
GCC_OPTIONS="-c -g -I./src/c/src $OPTIM_OPTIONS -o"

gcc -fPIC -I/usr/local/opt/openblas/include $GCC_OPTIONS  src/c/bin/fista.o src/c/src/fista.c
gcc -shared -g -lopenblas -L/usr/local/opt/openblas/lib $OPTIM_OPTIONS  -o src/c/bin/fista.so src/c/bin/fista.o

rm -f src/c/bin/*.o
rm -rf src/c/bin/*.dSYM