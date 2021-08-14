#include "tests_slab_random.hpp"
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <iostream>
#include <curand_kernel.h>

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

__global__ 
void generateUniform(cuFFT<float>::C_t* array, int N, unsigned long long seed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        curandState state;
        curand_init(seed, 2*i, 0, &state);
        array[i].x = curand_uniform(&state);
        curand_init(seed, 2*i+1, 0, &state);
        array[i].y = curand_uniform(&state);
    }
}

__global__ 
void generateUniformDouble(cuFFT<double>::C_t* array, int N, unsigned long long seed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        curandState state;
        curand_init(seed, 2*i, 0, &state);
        array[i].x = curand_uniform_double(&state);
        curand_init(seed, 2*i+1, 0, &state);
        array[i].y = curand_uniform_double(&state);
    }
}

template<typename T>
struct Generator {
    static decltype(generateUniform)* generateUniformComplex;
};
template<typename T> decltype(generateUniform)* Generator<T>::generateUniformComplex = generateUniform;

template<>
struct Generator<double> {
    static decltype(generateUniformDouble)* generateUniformComplex;
};
decltype(generateUniformDouble)* Generator<double>::generateUniformComplex = generateUniformDouble;

template<typename T>
int Tests_Slab_Random<T>::initializeRandArrayComplex(void* in_d, size_t N){
    using R_t = typename cuFFT<T>::R_t;
    using C_t = typename cuFFT<T>::C_t;

    C_t *complex = cuFFT<T>::complex(in_d);

    //set seed of generator
    unsigned long long seed = (unsigned long long) clock();
    
    Generator<T>::generateUniformComplex<<<N/1024+1, 1024>>>(complex, N, seed);

    Random_Tests<T>::scaleUniformComplexArray<<<N/1024+1, 1024>>>(complex, 255, N);

    CUDA_CALL(cudaDeviceSynchronize());

    return 0;
}

template<typename T>
int Tests_Slab_Random<T>::initializeRandArray(void* in_d, size_t N1){
    using R_t = typename cuFFT<T>::R_t;
    using C_t = typename cuFFT<T>::C_t;

    curandGenerator_t gen;
    R_t *real = cuFFT<T>::real(in_d);

    //create pseudo-random generator
    CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    //set seed of generator
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, (unsigned long long) clock()));
    CURAND_CALL(Random_Tests<T>::generateUniform(gen, real, N1*Ny*Nz));

    Random_Tests<T>::scaleUniformArray<<<(N1*Ny*Nz)/1024+1, 1024>>>(real, 255, N1*Ny*Nz);

    CURAND_CALL(curandDestroyGenerator(gen));

    return 0;
}

template class Tests_Slab_Random<float>;
template class Tests_Slab_Random<double>;