#! /bin/bash


MAT_FOLDER=$1
MAT_FILE=$2


echo "----- Building -----"

export CUDACXX=$(which nvcc)

# Create build folder
mkdir -p build && cd build

# Configure
cmake -DCMAKE_BUILD_TYPE=Release ..

cmake --build . --config Release -- -j4

echo "impl,time,matrix" > sptrsv_supernodal_bench

while read line; do
  mat="$MAT_FOLDER/$line"
  echo $mat
  ./bench  $mat >> sptrsv_supernodal_bench
done < $MAT_FILE