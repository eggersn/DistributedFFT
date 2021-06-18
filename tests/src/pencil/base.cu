#include "tests_pencil_random.hpp"
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

template<typename T>
int Tests_Pencil_Random<T>::initializeRandArray(void* in_d, size_t N1, size_t N2){
    using R_t = typename cuFFT<T>::R_t;
    using C_t = typename cuFFT<T>::C_t;

    curandGenerator_t gen;
    R_t *real = cuFFT<T>::real(in_d);

    //create pseudo-random generator
    CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    //set seed of generator
    // CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL));
    //get poisson samples
    CURAND_CALL(Random_Tests<T>::generateUniform(gen, real, N1*N2*Nz));

    Random_Tests<T>::scaleUniformArray<<<(N1*N2*Nz)/1024+1, 1024>>>(real, 255, N1*N2*Nz);

    return 0;
}

template class Tests_Pencil_Random<float>;
template class Tests_Pencil_Random<double>;