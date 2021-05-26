#nvcc -I./include -I/usr/include -L/usr/lib64 -L/usr/lib/openmpi -lmpi -lcufft -lmpi_cxx -lcurand -rdc=true src/mpicufft.cpp src/mpicufftslab.cpp tests/random.cu -o main

cmake_minimum_required(VERSION 3.10)
project(FFT LANGUAGES CUDA CXX)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

include_directories(./include)

find_package(CUDAToolkit REQUIRED CUDA::cudart CUDA::cufft CUDA::curand CUDA::cublas)

#MPI
find_package(MPI REQUIRED)

# create library and link them
add_library(mpicufft SHARED src/mpicufft.cpp src/mpicufft_slab.cpp src/mpicufft_slab_z_then_yx.cpp src/mpicufft_slab_y_then_zx.cpp src/mpicufft_pencil.cpp)
add_library(timer SHARED src/timer.cpp)
add_executable(slab_random tests/slab/random.cu)
add_executable(slab_random_dist tests/slab/random_dist.cu)
add_executable(slab_random_dist_z_then_yx tests/slab/random_dist_z_then_yx.cu)
add_executable(slab_random_dist_y_then_zx tests/slab/random_dist_y_then_zx.cu)

add_executable(pencil_random tests/pencil/random.cu)
add_executable(pencil_random_dist1 tests/pencil/random_dist_1D.cu)
add_executable(pencil_random_dist2 tests/pencil/random_dist_2D.cu)
add_executable(pencil_random_dist3 tests/pencil/random_dist_3D.cu)

target_link_libraries(slab_random PUBLIC mpicufft timer)
target_link_libraries(slab_random_dist PUBLIC mpicufft timer)
target_link_libraries(slab_random_dist_z_then_yx PUBLIC mpicufft timer)
target_link_libraries(slab_random_dist_y_then_zx PUBLIC mpicufft timer)

target_link_libraries(pencil_random PUBLIC mpicufft timer)
target_link_libraries(pencil_random_dist1 PUBLIC mpicufft timer)
target_link_libraries(pencil_random_dist2 PUBLIC mpicufft timer)
target_link_libraries(pencil_random_dist3 PUBLIC mpicufft timer)

#CUDA 
set_property(TARGET mpicufft slab_random slab_random_dist slab_random_dist_z_then_yx slab_random_dist_y_then_zx pencil_random pencil_random_dist1 pencil_random_dist2 pencil_random_dist3 PROPERTY CUDA_ARCHITECTURES 50 61)

#MPI
target_link_libraries(mpicufft PUBLIC MPI::MPI_CXX)
target_link_libraries(slab_random PUBLIC MPI::MPI_CXX)
target_link_libraries(slab_random_dist PUBLIC MPI::MPI_CXX)
target_link_libraries(slab_random_dist_z_then_yx PUBLIC MPI::MPI_CXX)
target_link_libraries(slab_random_dist_y_then_zx PUBLIC MPI::MPI_CXX)

target_link_libraries(pencil_random PUBLIC MPI::MPI_CXX)
target_link_libraries(pencil_random_dist1 PUBLIC MPI::MPI_CXX)
target_link_libraries(pencil_random_dist2 PUBLIC MPI::MPI_CXX)
target_link_libraries(pencil_random_dist3 PUBLIC MPI::MPI_CXX)

#CUDA
target_link_libraries(mpicufft PUBLIC CUDA::cudart CUDA::cufft)
target_link_libraries(slab_random PUBLIC CUDA::cudart CUDA::curand CUDA::cublas)
target_link_libraries(slab_random_dist PUBLIC CUDA::cudart CUDA::curand CUDA::cublas)
target_link_libraries(slab_random_dist_z_then_yx PUBLIC CUDA::cudart CUDA::curand CUDA::cublas)
target_link_libraries(slab_random_dist_y_then_zx PUBLIC CUDA::cudart CUDA::curand CUDA::cublas)

target_link_libraries(pencil_random PUBLIC CUDA::cudart CUDA::curand CUDA::cublas)
target_link_libraries(pencil_random_dist1 PUBLIC CUDA::cudart CUDA::curand CUDA::cublas)
target_link_libraries(pencil_random_dist2 PUBLIC CUDA::cudart CUDA::curand CUDA::cublas)
target_link_libraries(pencil_random_dist3 PUBLIC CUDA::cudart CUDA::curand CUDA::cublas)





