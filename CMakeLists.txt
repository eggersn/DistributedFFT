#nvcc -I./include -I/usr/include -L/usr/lib64 -L/usr/lib/openmpi -lmpi -lcufft -lmpi_cxx -lcurand -rdc=true src/mpicufft.cpp src/mpicufftslabs.cpp tests/random.cu -o main

cmake_minimum_required(VERSION 3.10)
project(FFT LANGUAGES CUDA CXX)

include_directories(./include)

find_package(CUDAToolkit REQUIRED CUDA::cudart CUDA::cufft CUDA::curand CUDA::cublas)

#MPI
find_package(MPI REQUIRED)

# create library and link them
add_library(mpicufft SHARED src/mpicufft.cpp src/mpicufftslabs.cpp)
add_executable(random tests/random.cu)
add_executable(random_dist tests/random_dist.cu)

target_link_libraries(random PUBLIC mpicufft)
target_link_libraries(random_dist PUBLIC mpicufft)

#CUDA 
set_property(TARGET mpicufft random random_dist PROPERTY CUDA_ARCHITECTURES 50)

#MPI
target_link_libraries(mpicufft PUBLIC MPI::MPI_CXX)
target_link_libraries(random PUBLIC MPI::MPI_CXX)
target_link_libraries(random_dist PUBLIC MPI::MPI_CXX)

#CUDA
target_link_libraries(mpicufft PUBLIC CUDA::cudart CUDA::cufft)
target_link_libraries(random PUBLIC CUDA::cudart CUDA::curand CUDA::cublas)
target_link_libraries(random_dist PUBLIC CUDA::cudart CUDA::curand CUDA::cublas)





