g++ $1.cpp -o build/$1 -O3 -march=native -mavx && time build/$1
