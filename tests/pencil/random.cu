#include "mpicufft_pencil.hpp"
#include "device_launch_parameters.h"
#include "cufft.hpp"
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

#define Nx 32
#define Ny 32
#define Nz 32

#define P1 2
#define P2 1

using R_t = typename cuFFT<float>::R_t;
using C_t = typename cuFFT<float>::C_t;

__global__ void scaleUniformArray( float* data_d, float factor, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n) {
        data_d[i] *= factor;
    }
}

int initializeRandArray(float* in_d, size_t N1, size_t N2){
    curandGenerator_t gen;

    //create pseudo-random generator
    CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    //set seed of generator
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL));
    //get poisson samples
    CURAND_CALL(curandGenerateUniform(gen, in_d, N1*N2*Nz));

    scaleUniformArray<<<(N1*N2*Nz)/1024+1, 1024>>>(in_d, 255, N1*Ny*Nz);

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

    size_t pidx_i = rank / P2;
    size_t pidx_j = rank % P2;
        
    //initialize MPIcuFFT
    MPIcuFFT_Pencil<float> mpicuFFT(MPI_COMM_WORLD, true);

    Pencil_Partition partition(P1, P2);
    GlobalSize global_size(Nx, Ny, Nz);
    mpicuFFT.initFFT(&global_size, &partition, true);

    // Allocate Memory
    Partition_Dimensions input_dim, transposed_dim, output_dim;
    mpicuFFT.getPartitionDimensions(input_dim, transposed_dim, output_dim);

    size_t out_size = std::max(input_dim.size_x[pidx_i]*input_dim.size_y[pidx_j]*(Nz/2+1), transposed_dim.size_x[pidx_i]*transposed_dim.size_y[0]*transposed_dim.size_z[pidx_j]);
    out_size = std::max(out_size, output_dim.size_x[0]*output_dim.size_y[pidx_i]*output_dim.size_z[pidx_j]);

    float *in_d, *out_d, *out_h;

    CUDA_CALL(cudaMalloc((void **)&in_d, input_dim.size_x[pidx_i]*input_dim.size_y[pidx_j]*Nz*sizeof(R_t)));
    CUDA_CALL(cudaMalloc((void **)&out_d, out_size*sizeof(C_t)));
    //allocate memory (host)
    out_h = (float *)calloc(out_size, sizeof(C_t));

    printf("outsize (%d, %d): %d", pidx_i, pidx_j, out_size);

    //random input
    initializeRandArray(in_d, input_dim.size_x[pidx_i], input_dim.size_y[pidx_j]);
    CUDA_CALL(cudaDeviceSynchronize());

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