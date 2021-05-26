#include "mpicufft_slab_y_then_zx.hpp"
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

#define Nx 128
#define Ny 128
#define Nz 128

#define ALLOW_CUDA_AWARE 1
#define CUDA_AWARE MPIX_Query_cuda_support() * ALLOW_CUDA_AWARE

using R_t = typename cuFFT<double>::R_t;
using C_t = typename cuFFT<double>::C_t;

__global__ void scaleUniformArray(R_t* data_d, R_t factor, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n) {
        data_d[i] *= factor;
    }
}

__global__ void difference(C_t* array1, C_t* array2, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n) {
        int z = i % (Nz/2+1);
        int y = ((i - z) / (Nz/2+1)) % Ny;
        int x = ((i-z)/(Nz/2+1) - y) / Ny;

        if (y < Ny/2+1) {
            int j = (x*(Ny/2+1)+y)*Nz+z;
            array1[i].x -= array2[j].x;
            array1[i].y -= array2[j].y;
        } else {
            int j = (((Nx-x)%Nx)*(Ny/2+1)+((Ny-y)%Ny))*Nz+((Nz-z)%Nz);
            array1[i].x -= array2[j].x;
            array1[i].y += array2[j].y;
        }
    }
}


int initializeRandArray(void* in_d){
    curandGenerator_t gen;
    R_t *real = cuFFT<double>::real(in_d);

    //create pseudo-random generator
    CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    //set seed of generator
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL));
    //get poisson samples
    CURAND_CALL(curandGenerateUniformDouble(gen, real, Nx*Ny*Nz));

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
    CUDA_CALL(cudaMalloc((void **)&res_d, Nx*(Ny/2+1)*Nz*sizeof(C_t)));
    
    if (CUDA_AWARE == 1){
        CUDA_CALL(cudaMalloc((void **)&send_ptr, Nx*Ny*Nz*sizeof(R_t)));
        CUDA_CALL(cudaMalloc((void **)&recv_ptr, Nx*(Ny/2+1)*Nz*sizeof(C_t)));
    } else {
        CUDA_CALL(cudaMallocHost((void **)&send_ptr, Nx*Ny*Nz*sizeof(R_t)));
        CUDA_CALL(cudaMallocHost((void **)&recv_ptr, Nx*(Ny/2+1)*Nz*sizeof(C_t)));
    }

    //random initialization of full Nx*Ny*Nz array
    initializeRandArray(in_d);

    //Copy input data to send-buffer and initialize cufft
    CUDA_CALL(cudaMemcpyAsync(send_ptr, in_d, Nx*Ny*Nz*sizeof(R_t), 
        CUDA_AWARE==1?cudaMemcpyDeviceToDevice:cudaMemcpyDeviceToHost));

    CUFFT_CALL(cufftCreate(&planR2C));
    CUFFT_CALL(cufftMakePlan3d(planR2C, Nx, Ny, Nz, cuFFT<double>::R2Ctype, &ws_r2c));


    //Distribute input data
    size_t N1 = Nx/world_size;
    size_t N2 = (Ny/2+1)/world_size;
    size_t send_count = 0;
    size_t recv_count = 0;
    std::vector<size_t> recv_counts;
    std::vector<size_t> ostarty;
    ostarty.push_back(0);
    for (int pidx = 0; pidx < world_size; pidx++){
        size_t Nxpidx = N1 + (pidx<Nx%world_size?1:0);
        size_t Nypidx = N2 + (pidx<(Ny/2+1)%world_size?1:0);
        recv_req[pidx] = MPI_REQUEST_NULL;
        send_req[pidx] = MPI_REQUEST_NULL;
        ostarty.push_back(ostarty[pidx]+Nypidx);

        //start non-blocking receive for distributed results (asynch to local fft computation)
        MPI_Irecv(&recv_ptr[recv_count], Nx*Nypidx*Nx*sizeof(C_t), MPI_BYTE, pidx, pidx, MPI_COMM_WORLD, &recv_req[pidx]);
        recv_counts.push_back(recv_count);
        recv_count += Nx*Nypidx*Nz;

        //start non-blocking send for input data
        MPI_Isend(&send_ptr[send_count], Nxpidx*Ny*Nz*sizeof(R_t), MPI_BYTE, pidx, pidx, MPI_COMM_WORLD, &send_req[pidx]);
        send_count += Nxpidx*Ny*Nz;
    }

    //wait till all input data has been distributed
    MPI_Waitall(world_size, send_req.data(), MPI_STATUSES_IGNORE);

    //compute local fft
    R_t *real    = cuFFT<double>::real(in_d);
    C_t *complex = cuFFT<double>::complex(out_d);
    
    CUFFT_CALL(cuFFT<double>::execR2C(planR2C, real, complex));
    CUDA_CALL(cudaDeviceSynchronize());

    CUBLAS_CALL(cublasCreate(&handle));

    int p;
    do {
        // recv_req contains one null handle (i.e. recv_req[pidx_i]) and P1-1 active handles
        // If all active handles are processed, Waitany will return MPI_UNDEFINED
        MPI_Waitany(world_size, recv_req.data(), &p, MPI_STATUSES_IGNORE);
        if (p == MPI_UNDEFINED)
            break;

        size_t osizey = N2 + (p<(Ny/2+1)%world_size?1:0);
        
        cudaMemcpy3DParms cpy_params = {0};
        cpy_params.srcPos = make_cudaPos(0, 0, 0);
        cpy_params.srcPtr = make_cudaPitchedPtr(&recv_ptr[recv_counts[p]], Nz*sizeof(C_t), Nz, osizey);
        cpy_params.dstPos = make_cudaPos(0, ostarty[p], 0);
        cpy_params.dstPtr = make_cudaPitchedPtr(res_d, Nz*sizeof(C_t), Nz, Ny/2+1);    
        cpy_params.extent = make_cudaExtent(Nz*sizeof(C_t), osizey, Nx);
        cpy_params.kind   = CUDA_AWARE==1 ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice;   
        
        CUDA_CALL(cudaMemcpy3DAsync(&cpy_params));
    } while (p != MPI_UNDEFINED);
    CUDA_CALL(cudaDeviceSynchronize());

    //compare difference
    double sum = 0;
    difference<<<Nx*(Ny/2+1)*Nz/1024+1, 1024>>>(complex, res_d, Nx*(Ny/2+1)*Nz);

    CUBLAS_CALL(cublasDzasum(handle, Nx*Ny*Nz, complex, 1, &sum));
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
    MPI_Request send_req;
    MPI_Request recv_req;

    size_t N1=Nx/world_size;
    size_t N2=(Ny/2+1)/world_size;
    if (rank < Nx%world_size)
        N1++;
    if (rank < (Ny/2+1)%world_size)
        N2++;

    R_t *in_d, *recv_ptr;
    C_t *out_d, *send_ptr;
    size_t out_size = std::max(N1*(Ny/2+1)*Nz, Nx*N2*Nz);

    //allocate memory (device)
    CUDA_CALL(cudaMalloc((void **)&in_d, N1*Ny*Nz*sizeof(R_t)));
    CUDA_CALL(cudaMalloc((void **)&out_d, out_size*sizeof(C_t)));
    
    if (CUDA_AWARE == 1){
        recv_ptr = in_d;
        send_ptr = out_d;
    } else {
        CUDA_CALL(cudaMallocHost((void **)&recv_ptr, N1*Ny*Nz*sizeof(R_t)));
        CUDA_CALL(cudaMallocHost((void **)&send_ptr, Nx*N2*Nz*sizeof(C_t)));
    }

    //receive input data via MPI
    MPI_Irecv(recv_ptr, N1*Ny*Nz*sizeof(R_t), MPI_BYTE, world_size, rank, MPI_COMM_WORLD, &recv_req);
    MPI_Wait(&recv_req, MPI_STATUSES_IGNORE);

    if (CUDA_AWARE == 0){
        CUDA_CALL(cudaMemcpyAsync(in_d, recv_ptr, N1*Ny*Nz*sizeof(R_t), cudaMemcpyHostToDevice));
        CUDA_CALL(cudaDeviceSynchronize());
    }

    //initialize MPIcuFFT
    MPIcuFFT_Slab_Y_Then_ZX<double> mpicuFFT(MPI_COMM_WORLD, CUDA_AWARE==1, world_size);
    
    GlobalSize global_size(Nx, Ny, Nz);
    mpicuFFT.initFFT(&global_size, true);

    //execute
    mpicuFFT.execR2C(out_d, in_d);

    if (CUDA_AWARE == 0){
        CUDA_CALL(cudaMemcpyAsync(send_ptr, out_d, Nx*N2*Nz*sizeof(C_t), cudaMemcpyDeviceToHost));
        CUDA_CALL(cudaDeviceSynchronize());
    }

    MPI_Isend(send_ptr, Nx*N2*Nz*sizeof(C_t), MPI_BYTE, world_size, rank, MPI_COMM_WORLD, &send_req);
    MPI_Wait(&send_req, MPI_STATUSES_IGNORE);
    
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

