#nvcc -I./include -I/usr/include -L/usr/lib64 -L/usr/lib/openmpi -lmpi -lcufft -lmpi_cxx -lcurand -rdc=true src/mpicufft.cpp src/mpicufftslab.cpp tests/random.cu -o main

cmake_minimum_required(VERSION 3.10)
project(FFT LANGUAGES CUDA CXX)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

include_directories(./include ./tests/include)

find_package(CUDAToolkit REQUIRED CUDA::cudart CUDA::cufft CUDA::curand CUDA::cublas)

#MPI
set(MPI_HOME ${MPI_HOME})
find_package(MPI REQUIRED)

# create library and link them

add_library(mpicufft SHARED src/mpicufft.cpp src/slab/default/mpicufft_slab.cpp src/slab/default/mpicufft_slab_opt1.cpp src/slab/z_then_yx/mpicufft_slab_z_then_yx.cpp src/slab/z_then_yx/mpicufft_slab_z_then_yx_opt1.cpp src/slab/y_then_zx/mpicufft_slab_y_then_zx.cpp src/pencil/mpicufft_pencil.cpp)
add_library(slab_tests SHARED tests/slab/base.cu tests/slab/random_dist.cu tests/slab/random_dist_y_then_zx.cu tests/slab/random_dist_z_then_yx.cu )
add_library(pencil_tests SHARED tests/pencil/base.cu tests/pencil/random_dist_1D.cu tests/pencil/random_dist_2D.cu tests/pencil/random_dist_3D.cu)
add_library(timer SHARED src/timer.cpp)
add_executable(tests_exec tests/base.cu tests/main.cpp)

set_property(TARGET timer PROPERTY CXX_STANDARD 20)
target_link_libraries(tests_exec PUBLIC mpicufft slab_tests pencil_tests timer)

#CUDA 
set_property(TARGET mpicufft slab_tests pencil_tests tests_exec PROPERTY CUDA_ARCHITECTURES 50 61)

#MPI
target_link_libraries(mpicufft PUBLIC MPI::MPI_CXX)
target_link_libraries(slab_tests PUBLIC MPI::MPI_CXX)
target_link_libraries(pencil_tests PUBLIC MPI::MPI_CXX)
target_link_libraries(timer PUBLIC MPI::MPI_CXX)

#CUDA
target_link_libraries(mpicufft PUBLIC CUDA::cudart CUDA::cufft)
target_link_libraries(slab_tests PUBLIC CUDA::cudart CUDA::curand CUDA::cublas)
target_link_libraries(pencil_tests PUBLIC CUDA::cudart CUDA::curand CUDA::cublas)





