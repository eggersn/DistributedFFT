#nvcc -I./include -I/usr/include -L/usr/lib64 -L/usr/lib/openmpi -lmpi -lcufft -lmpi_cxx -lcurand -rdc=true src/mpicufft.cpp src/mpicufftslab.cpp tests/random.cu -o main

cmake_minimum_required(VERSION 3.10)
project(Distributed_FFT_Comparison LANGUAGES CUDA CXX)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

include_directories(include tests/include)

#MPI
set(MPI_HOME ${MPI_HOME})
set(MPI_CXX_COMPILER_FLAGS "-g")
find_package(MPI REQUIRED)

# ********************* MPI_cuFFT Library ********************* #

set(MPICUFFT_FILES src/mpicufft.cpp)
set(SLAB_FILES src/slab/default/mpicufft_slab.cpp src/slab/default/mpicufft_slab_opt1.cpp src/slab/z_then_yx/mpicufft_slab_z_then_yx.cpp src/slab/z_then_yx/mpicufft_slab_z_then_yx_opt1.cpp src/slab/y_then_zx/mpicufft_slab_y_then_zx.cpp)
set(PENCIL_FILES src/pencil/mpicufft_pencil.cpp src/pencil/mpicufft_pencil_opt1.cpp)
set(TIMER_FILES src/timer.cpp)

# create library and link them
add_library(mpicufft SHARED ${MPICUFFT_FILES})
add_library(slab_decomp SHARED ${SLAB_FILES})
add_library(pencil_decomp SHARED ${PENCIL_FILES})
add_library(timer SHARED ${TIMER_FILES})

set_property(TARGET timer PROPERTY CXX_STANDARD 11)

if(${CMAKE_VERSION} VERSION_LESS "3.18.0") 
    set_source_files_properties(${MPICUFFT_FILES} ${SLAB_FILES} ${PENCIL_FILES} PROPERTIES LANGUAGE CUDA)

    target_link_libraries(mpicufft PUBLIC MPI::MPI_CXX -lcudart -lcufft)
    target_link_libraries(slab_decomp PUBLIC MPI::MPI_CXX -lcudart -lcufft)
    target_link_libraries(pencil_decomp PUBLIC MPI::MPI_CXX -lcudart -lcufft)
    target_link_libraries(timer PUBLIC MPI::MPI_CXX)
else()
    find_package(CUDAToolkit REQUIRED CUDA::cudart CUDA::cufft CUDA::curand CUDA::cublas)
    # requires cmake 3.18
    set_property(TARGET mpicufft slab_decomp pencil_decomp PROPERTY CUDA_ARCHITECTURES 50 61 80 86)

    # link CUDA libraries
    target_link_libraries(mpicufft PUBLIC CUDA::cudart CUDA::cufft)
    target_link_libraries(slab_decomp PUBLIC CUDA::cudart CUDA::cufft)
    target_link_libraries(pencil_decomp PUBLIC CUDA::cudart CUDA::cufft)

    # link MPI libraries
    target_link_libraries(mpicufft PUBLIC MPI::MPI_CXX)
    target_link_libraries(slab_decomp PUBLIC MPI::MPI_CXX)
    target_link_libraries(pencil_decomp PUBLIC MPI::MPI_CXX)
    target_link_libraries(timer PUBLIC MPI::MPI_CXX)
endif()

# link libraries
target_link_libraries(slab_decomp PUBLIC mpicufft timer)
target_link_libraries(pencil_decomp PUBLIC mpicufft timer)

# ********************* MPI_cuFFT Tests ********************* #

set(TEST_BASE_FILES tests/src/base.cu )
set(SLABTEST_FILES tests/src/slab/base.cu tests/src/slab/random_dist_default.cu tests/src/slab/random_dist_y_then_zx.cu tests/src/slab/random_dist_z_then_yx.cu)
set(PENCILTEST_FILES tests/src/pencil/base.cu tests/src/pencil/random_dist_1D.cu tests/src/pencil/random_dist_2D.cu tests/src/pencil/random_dist_3D.cu)
set(REFERENCETEST_FILES tests/src/reference/reference.cu)

set(EXEC_REF_FILES tests/src/reference/main.cpp)
set(EXEC_SLAB_FILES tests/src/slab/main.cpp)
set(EXEC_PENCIL_FILES tests/src/pencil/main.cpp)

# create library and link them
add_library(test_base SHARED ${TEST_BASE_FILES})
add_library(slab_tests SHARED ${SLABTEST_FILES})
add_library(pencil_tests SHARED ${PENCILTEST_FILES})
add_library(reference_tests SHARED ${REFERENCETEST_FILES})

add_executable(reference ${EXEC_REF_FILES})
add_executable(slab ${EXEC_SLAB_FILES})
add_executable(pencil ${EXEC_PENCIL_FILES})

if(${CMAKE_VERSION} VERSION_LESS "3.18.0") 
    set_source_files_properties(tests/src/main.cpp PROPERTIES LANGUAGE CUDA)

    target_link_libraries(test_base PUBLIC MPI::MPI_CXX -lcudart -lcurand -lcublas)
    target_link_libraries(slab_tests PUBLIC MPI::MPI_CXX -lcudart -lcurand -lcublas)
    target_link_libraries(pencil_tests PUBLIC MPI::MPI_CXX -lcudart -lcurand -lcublas)
    target_link_libraries(timer PUBLIC MPI::MPI_CXX)
else()
    # requires cmake 3.17
    find_package(CUDAToolkit REQUIRED CUDA::cudart CUDA::cufft CUDA::curand CUDA::cublas)
    # requires cmake 3.18
    set_property(TARGET test_base slab_tests pencil_tests reference_tests PROPERTY CUDA_ARCHITECTURES 50 61 80 86)
    
    # link CUDA libraries
    target_link_libraries(test_base PUBLIC CUDA::cudart CUDA::curand CUDA::cublas)
    target_link_libraries(slab_tests PUBLIC CUDA::cudart CUDA::cufft CUDA::curand CUDA::cublas)
    target_link_libraries(pencil_tests PUBLIC CUDA::cudart CUDA::cufft CUDA::curand CUDA::cublas)
    target_link_libraries(reference_tests PUBLIC CUDA::cudart CUDA::cufft CUDA::curand CUDA::cublas)
    
    # link MPI libraries
    target_link_libraries(slab_tests PUBLIC MPI::MPI_CXX)
    target_link_libraries(pencil_tests PUBLIC MPI::MPI_CXX)
    target_link_libraries(reference_tests PUBLIC MPI::MPI_CXX)
endif()

# link libraries
target_link_libraries(slab_tests PUBLIC slab_decomp test_base)
target_link_libraries(pencil_tests PUBLIC pencil_decomp test_base)
target_link_libraries(reference_tests PUBLIC mpicufft test_base timer)

target_link_libraries(reference PUBLIC reference_tests)
target_link_libraries(slab PUBLIC slab_tests)
target_link_libraries(pencil PUBLIC pencil_tests)







