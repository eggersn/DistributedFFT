#include "mpicufft_slab.hpp"
#include "device_launch_parameters.h"
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand.h>
#include <unistd.h>
#include <iostream>

#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)
#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)

#define Nx 64
#define Ny 64
#define Nz 64

__global__ void scaleUniformArray( float* data_d, float factor, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n) {
        data_d[i] *= factor;
    }
}

int initializeRandArray(float* in_d, size_t N1){
    curandGenerator_t gen;

    //create pseudo-random generator
    CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    //set seed of generator
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL));
    //get poisson samples
    CURAND_CALL(curandGenerateUniform(gen, in_d, N1*Ny*Nz));

    scaleUniformArray<<<(N1*Ny*Nz)/1024+1, 1024>>>(in_d, 255, N1*Ny*Nz);

    return 0;
}

int main() {       
    //initialize MPI
    MPI_Init(NULL, NULL);

    //number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    //get global rank
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    size_t N1=Nx/world_size;
    size_t N2=Ny/world_size;
    if (rank < Nx%world_size)
        N1++;
    if (rank < Ny%world_size)
        N2++;

    float *in_d, *out_d, *out_h;
    size_t out_size = 2*std::max(N1*Ny*(Nz/2+1), Nx*N2*(Nz/2+1));

    //allocate memory (device)
    CUDA_CALL(cudaMalloc((void **)&in_d, N1*Ny*Nz*sizeof(float)));
    CUDA_CALL(cudaMalloc((void **)&out_d, out_size*sizeof(float)));
    //allocate memory (host)
    out_h = (float *)calloc(out_size, sizeof(float));
    
    //random input
    initializeRandArray(in_d, N1);
    
    //initialize MPIcuFFT
    MPIcuFFT_Slab<float> mpicuFFT(MPI_COMM_WORLD, true);

    GlobalSize global_size(Nx, Ny, Nz);
    mpicuFFT.initFFT(&global_size, true);

    //execute
    mpicuFFT.execR2C(out_d, in_d);

    CUDA_CALL(cudaMemcpy(out_h, out_d, out_size*sizeof(float), cudaMemcpyDeviceToHost));

    //do stuff with out_h

    //finalize
    MPI_Finalize();

    CUDA_CALL(cudaFree(in_d));
    CUDA_CALL(cudaFree(out_d));
    free(out_h);

    return 0;
}