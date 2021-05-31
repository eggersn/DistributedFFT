#include "tests_slab_random.hpp"
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <iostream>

#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)
#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) {    \
    printf("Error at %s:%d\n",__FILE__,__LINE__);               \
    return EXIT_FAILURE;}} while(0)
#define CUBLAS_CALL(x) do { if((x)!=CUBLAS_STATUS_SUCCESS) {    \
    printf("Error %d at %s:%d\n",x,__FILE__,__LINE__);          \
    return EXIT_FAILURE;}} while(0)
#define CUFFT_CALL(x) do { if((x)!=CUFFT_SUCCESS) {             \
    printf("Error %d at %s:%d\n",x,__FILE__,__LINE__);          \
    return EXIT_FAILURE;}} while(0)

__global__ void scaleUniformArrayFloat(cuFFT<float>::R_t* data_d, cuFFT<float>::R_t factor, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n) {
        data_d[i] *= factor;
    }
}

__global__ void scaleUniformArrayDouble(cuFFT<double>::R_t* data_d, cuFFT<double>::R_t factor, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n) {
        data_d[i] *= factor;
    }
}

decltype(curandGenerateUniformDouble)* Random_Tests<double>::generateUniform = curandGenerateUniformDouble;
decltype(cublasDzasum)* Random_Tests<double>::cublasSum = cublasDzasum;
decltype(scaleUniformArrayDouble)* Random_Tests<double>::scaleUniformArray = scaleUniformArrayDouble;

template<typename T>
int Tests_Slab_Random<T>::initializeRandArray(void* in_d){
    using R_t = typename cuFFT<T>::R_t;
    using C_t = typename cuFFT<T>::C_t;

    curandGenerator_t gen;
    R_t *real = cuFFT<T>::real(in_d);

    //create pseudo-random generator
    CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    //set seed of generator
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL));
    //get poisson samples
    CURAND_CALL(Random_Tests<T>::generateUniform(gen, real, Nx*Ny*Nz));

    Random_Tests<T>::scaleUniformArray<<<(Nx*Ny*Nz)/1024+1, 1024>>>(real, 255, Nx*Ny*Nz);

    return 0;
}

template<typename T>
int Tests_Slab_Random<T>::run(int opt) {      
    //initialize MPI
    MPI_Init(NULL, NULL);

    //number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    world_size--;

    //get global rank
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == world_size){
        this->coordinate(world_size);
    } else{
        this->compute(rank, world_size, opt);
    }
    
    //finalize
    MPI_Finalize();

    return 0;
}

template class Tests_Slab_Random<float>;
template class Tests_Slab_Random<double>;