#include "mpicufft_slabs.hpp"
#include "cufft.hpp"
#include "mpi.h"
#include "mpi-ext.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include "cublas_v2.h"
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

#define Nx 32
#define Ny 32
#define Nz 32

#define ALLOW_CUDA_AWARE 1
#define CUDA_AWARE MPIX_Query_cuda_support() * ALLOW_CUDA_AWARE

using R_t = typename cuFFT<float>::R_t;
using C_t = typename cuFFT<float>::C_t;

__global__ void scaleUniformArray(R_t* data_d, R_t factor, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n) {
        data_d[i] *= factor;
    }
}

__global__ void difference(C_t* array1, C_t* array2, int n, int N2, int N2_mod){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n) {
        int x = i / (Ny * (Nz/2+1));
        int y = (i / (Nz/2+1)) % Ny;
        int z = i % (Nz/2+1);

        int y_num = y / N2;
        int y_start = 0, y_pidx = 0;
        if (y_num < N2_mod){
            y_start = (N2+1)*y_num;
            y_pidx = N2+1;
        } else{
            y_start = N2*y_num + N2_mod;
            y_pidx = N2;
        } 
        
        int index = y_start*(Nz/2+1)*Nx + x*y_pidx*(Nz/2+1) + (y - y_start)*(Nz/2+1) + z;

        array1[i].x -= array2[index].x;
        array1[i].y -= array2[index].y;
    }
}


int initializeRandArray(void* in_d){
    curandGenerator_t gen;
    R_t *real = cuFFT<float>::real(in_d);

    //create pseudo-random generator
    CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    //set seed of generator
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL));
    //get poisson samples
    CURAND_CALL(curandGenerateUniform(gen, real, Nx*Ny*Nz));

    scaleUniformArray<<<(Nx*Ny*Nz)/1024+1, 1024>>>(real, 255, Nx*Ny*Nz);

    return 0;
}

int coordinate(int world_size){
    std::vector<MPI_Request> send_req;
    std::vector<MPI_Request> recv_req;

    R_t *in_d, *send_ptr;
    C_t *out_d, *recv_ptr, *res_d;

    size_t ws_r2c;

    cufftHandle planR2C;
    cublasHandle_t handle;

    send_req.resize(world_size, MPI_REQUEST_NULL);
    recv_req.resize(world_size, MPI_REQUEST_NULL);

    //allocate memory (device)
    CUDA_CALL(cudaMalloc((void **)&in_d, Nx*Ny*Nz*sizeof(R_t)));
    CUDA_CALL(cudaMalloc((void **)&out_d, Nx*Ny*Nz*sizeof(C_t)));
    
    if (CUDA_AWARE == 1){
        CUDA_CALL(cudaMalloc((void **)&send_ptr, Nx*Ny*Nz*sizeof(R_t)));
        CUDA_CALL(cudaMalloc((void **)&recv_ptr, Nx*Ny*(Nz/2+1)*sizeof(C_t)));
        res_d = recv_ptr;
    } else {
        CUDA_CALL(cudaMallocHost((void **)&send_ptr, Nx*Ny*Nz*sizeof(R_t)));
        CUDA_CALL(cudaMallocHost((void **)&recv_ptr, Nx*Ny*(Nz/2+1)*sizeof(C_t)));
        CUDA_CALL(cudaMalloc((void **)&res_d, Nx*Ny*(Nz/2+1)*sizeof(C_t)));
    }

    //random initialization of full Nx*Ny*Nz array
    initializeRandArray(in_d);

    //Copy input data to send-buffer and initialize cufft
    CUDA_CALL(cudaMemcpyAsync(send_ptr, in_d, Nx*Ny*Nz*sizeof(R_t), 
        CUDA_AWARE==1?cudaMemcpyDeviceToDevice:cudaMemcpyDeviceToHost));

    CUFFT_CALL(cufftCreate(&planR2C));
    CUFFT_CALL(cufftSetAutoAllocation(planR2C, 0));
    CUFFT_CALL(cufftMakePlan3d(planR2C, Nx, Ny, Nz, cuFFT<float>::R2Ctype, &ws_r2c));

    CUDA_CALL(cudaDeviceSynchronize());
    CUFFT_CALL(cufftSetWorkArea(planR2C, in_d));

    //Distribute input data
    size_t N1 = Nx/world_size;
    size_t N2 = Ny/world_size;
    size_t send_count = 0;
    size_t recv_count = 0;
    for (int pidx = 0; pidx < world_size; pidx++){
        size_t Nxpidx = N1 + (pidx<Nx%world_size?1:0);
        size_t Nypidx = N2 + (pidx<Ny%world_size?1:0);
        recv_req[pidx] = MPI_REQUEST_NULL;
        send_req[pidx] = MPI_REQUEST_NULL;

        //start non-blocking receive for distributed results (asynch to local fft computation)
        MPI_Irecv(&recv_ptr[recv_count], Nx*Nypidx*(Nz/2+1)*sizeof(C_t), MPI_BYTE, pidx, pidx, MPI_COMM_WORLD, &recv_req[pidx]);
        recv_count += Nx*Nypidx*(Nz/2+1);

        //start non-blocking send for input data
        MPI_Isend(&send_ptr[send_count], Nxpidx*Ny*Nz*sizeof(R_t), MPI_BYTE, pidx, pidx, MPI_COMM_WORLD, &send_req[pidx]);
        send_count += Nxpidx*Ny*Nz;
    }

    //wait till all input data has been distributed
    MPI_Waitall(world_size, send_req.data(), MPI_STATUSES_IGNORE);

    //compute local fft
    R_t *real    = cuFFT<float>::real(in_d);
    C_t *complex = cuFFT<float>::complex(out_d);

    CUFFT_CALL(cuFFT<float>::execR2C(planR2C, real, complex));
    CUDA_CALL(cudaDeviceSynchronize());

    CUBLAS_CALL(cublasCreate(&handle));

    //wait till distributed results are received
    MPI_Waitall(world_size, recv_req.data(), MPI_STATUSES_IGNORE);

    if (CUDA_AWARE==0){ //received data has to be copied to gpu
        CUDA_CALL(cudaMemcpyAsync(res_d, recv_ptr, Nx*Ny*(Nz/2+1)*sizeof(C_t), cudaMemcpyHostToDevice));
        CUDA_CALL(cudaDeviceSynchronize());
    }

    //compare difference
    int N2_mod = Ny % world_size;
    difference<<<(Nx*Ny*(Nz/2+1))/1024+1, 1024>>>(complex, res_d, Nx*Ny*(Nz/2+1), N2, N2_mod);

    float sum = 0;
    CUBLAS_CALL(cublasScasum(handle, Nx*Ny*(Nz/2+1), complex, 1, &sum));
    CUBLAS_CALL(cublasDestroy(handle));

    std::cout << "Result " << sum << std::endl;

    CUFFT_CALL(cufftDestroy(planR2C));

    CUDA_CALL(cudaFree(in_d));
    CUDA_CALL(cudaFree(out_d));
    if (CUDA_AWARE == 0){
        CUDA_CALL(cudaFree(res_d));
    } 

    return 0;
}

int compute(int rank, int world_size){
    std::vector<MPI_Request> send_req;
    std::vector<MPI_Request> recv_req;

    size_t N1=Nx/world_size;
    size_t N2=Ny/world_size;
    if (rank < Nx%world_size)
        N1++;
    if (rank < Ny%world_size)
        N2++;

    send_req.resize(1, MPI_REQUEST_NULL);
    recv_req.resize(1, MPI_REQUEST_NULL);

    R_t *in_d, *recv_ptr;
    C_t *out_d, *send_ptr;
    size_t out_size = std::max(N1*Ny*(Nz/2+1), Nx*N2*(Nz/2+1));

    //allocate memory (device)
    CUDA_CALL(cudaMalloc((void **)&in_d, N1*Ny*Nz*sizeof(R_t)));
    CUDA_CALL(cudaMalloc((void **)&out_d, out_size*sizeof(C_t)));
    
    if (CUDA_AWARE == 1){
        recv_ptr = in_d;
        send_ptr = out_d;
    } else {
        CUDA_CALL(cudaMallocHost((void **)&recv_ptr, N1*Ny*Nz*sizeof(R_t)));
        CUDA_CALL(cudaMallocHost((void **)&send_ptr, Nx*N2*(Nz/2+1)*sizeof(C_t)));
    }

    //receive input data via MPI
    MPI_Irecv(recv_ptr, N1*Ny*Nz*sizeof(R_t), MPI_BYTE, world_size, rank, MPI_COMM_WORLD, &recv_req[0]);
    MPI_Wait(&recv_req[0], MPI_STATUSES_IGNORE);

    if (CUDA_AWARE == 0){
        CUDA_CALL(cudaMemcpyAsync(in_d, recv_ptr, N1*Ny*Nz*sizeof(R_t), cudaMemcpyHostToDevice));
        CUDA_CALL(cudaDeviceSynchronize());
    }

    //initialize MPIcuFFT
    MPIcuFFT_Slabs<float> mpicuFFT(MPI_COMM_WORLD, CUDA_AWARE==1, world_size);
    
    GlobalSize global_size(Nx, Ny, Nz);
    mpicuFFT.initFFT(&global_size, true);

    //execute
    mpicuFFT.execR2C(out_d, in_d);

    if (CUDA_AWARE == 0){
        CUDA_CALL(cudaMemcpyAsync(send_ptr, out_d, Nx*N2*(Nz/2+1)*sizeof(C_t), cudaMemcpyDeviceToHost));
        CUDA_CALL(cudaDeviceSynchronize());
    }

    MPI_Isend(send_ptr, Nx*N2*(Nz/2+1)*sizeof(C_t), MPI_BYTE, world_size, rank, MPI_COMM_WORLD, &send_req[0]);
    MPI_Wait(&send_req[0], MPI_STATUSES_IGNORE);
    
    CUDA_CALL(cudaFree(in_d));
    CUDA_CALL(cudaFree(out_d));

    return 0;
}

int main() {      
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
        coordinate(world_size);
    } else{
        compute(rank, world_size);
    }
    
    //finalize
    MPI_Finalize();

    return 0;
}

