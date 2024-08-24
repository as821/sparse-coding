# compile C files into shared library. call from top-level directory of repository
OPTIM_OPTIONS="-O3 -fno-math-errno -fno-trapping-math -march=native"
GCC_OPTIONS="-c -g -I./src/c/src $OPTIM_OPTIONS -o"

gcc -fPIC -I/usr/local/opt/openblas/include $GCC_OPTIONS  src/c/bin/test.o src/c/src/test.c
gcc -shared -g -lopenblas -L/usr/local/opt/openblas/lib $OPTIM_OPTIONS  -o src/c/bin/test.so src/c/bin/test.o

rm -f src/c/bin/*.o
rm -rf src/c/bin/*.dSYM