#include "mpicufft_pencil.hpp"
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
    printf("Error %d at %s:%d\n",x,__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)
#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) {    \
    printf("Error %d at %s:%d\n",x,__FILE__,__LINE__);               \
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

#define P1 4
#define P2 4
#define PartOfCluster 0

#define ALLOW_CUDA_AWARE 0
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
    if (i < n) {
        array1[i].x -= array2[i].x;
        array1[i].y -= array2[i].y;
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

int coordinate(int world_size) {
    std::vector<MPI_Request> send_req;
    std::vector<MPI_Request> recv_req;
    
    R_t *in_d, *send_ptr;
    C_t *out_d, *recv_ptr, *res_d;
    
    Partition_Dimensions input_dim;
    Partition_Dimensions transposed_dim;
    Partition_Dimensions output_dim;

    send_req.resize(world_size, MPI_REQUEST_NULL);
    recv_req.resize(world_size, MPI_REQUEST_NULL);

    // Determine all Partition_Dimensions
    // input_dim:
    input_dim.size_x.resize(P1, Nx/P1);
    for (int i = 0; i < Nx%P1; i++)
        input_dim.size_x[i]++;
    input_dim.size_y.resize(P2, Ny/P2);
    for (int j = 0; j < Ny%P2; j++)
        input_dim.size_y[j]++;
    input_dim.size_z.resize(1, Nz);
    input_dim.computeOffsets();
    // transposed_dim:
    transposed_dim.size_x = input_dim.size_x;
    transposed_dim.size_y.resize(1, Ny);
    transposed_dim.size_z.resize(P2, (Nz/2+1)/P2);
    for (int k = 0; k < (Nz/2+1)%P2; k++)
        transposed_dim.size_z[k]++;
    transposed_dim.computeOffsets();
    // output_dim:
    output_dim.size_x.resize(1, Nx);
    output_dim.size_y.resize(P1, Ny/P1);
    for (int j = 0; j < Ny%P1; j++)
        output_dim.size_y[j]++;
    output_dim.size_z = transposed_dim.size_z;
    output_dim.computeOffsets();

    // Generate random input data for each partition
    // Allocate memory (device)
    CUDA_CALL(cudaMalloc((void **)&in_d, Nx*Ny*Nz*sizeof(R_t)));
    CUDA_CALL(cudaMalloc((void **)&out_d, Nx*Ny*(Nz/2+1)*sizeof(C_t)));
    CUDA_CALL(cudaMalloc((void **)&res_d, Nx*Ny*(Nz/2+1)*sizeof(C_t)));
    
    if (CUDA_AWARE == 1){
        CUDA_CALL(cudaMalloc((void **)&send_ptr, Nx*Ny*Nz*sizeof(R_t)));
        CUDA_CALL(cudaMalloc((void **)&recv_ptr, Nx*Ny*(Nz/2+1)*sizeof(C_t)));
    } else {
        CUDA_CALL(cudaMallocHost((void **)&send_ptr, Nx*Ny*Nz*sizeof(R_t)));
        CUDA_CALL(cudaMallocHost((void **)&recv_ptr, Nx*Ny*(Nz/2+1)*sizeof(C_t)));
    }

    //random initialization of full Nx*Ny*Nz array
    initializeRandArray(in_d);

    std::vector<size_t> recv_counts;
    size_t recv_count = 0;
    size_t send_count = 0;
    for (size_t p_i = 0; p_i < P1; p_i++){
        for (size_t p_j = 0; p_j < P2; p_j++){
            cudaMemcpy3DParms cpy_params = {0};
            cpy_params.srcPos = make_cudaPos(0, input_dim.start_y[p_j], input_dim.start_x[p_i]);
            cpy_params.srcPtr = make_cudaPitchedPtr(in_d, Nz*sizeof(R_t), Nz, Ny);
            cpy_params.dstPos = make_cudaPos(0, 0, 0);
            cpy_params.dstPtr = make_cudaPitchedPtr(&send_ptr[send_count], Nz*sizeof(R_t), Nz, input_dim.size_y[p_j]);
            cpy_params.extent = make_cudaExtent(Nz*sizeof(R_t), input_dim.size_y[p_j], input_dim.size_x[p_i]);
            cpy_params.kind   = CUDA_AWARE==1 ? cudaMemcpyDeviceToDevice : cudaMemcpyDeviceToHost;
            
            CUDA_CALL(cudaMemcpy3DAsync(&cpy_params));
            CUDA_CALL(cudaDeviceSynchronize());
            
            recv_counts.push_back(recv_count);

            //start non-blocking receive for distributed results (asynch to local fft computation)
            MPI_Irecv(&recv_ptr[recv_count], output_dim.size_x[0]*output_dim.size_y[p_i]*output_dim.size_z[p_j]*sizeof(C_t), 
            MPI_BYTE, p_i*P2+p_j, world_size, MPI_COMM_WORLD, &recv_req[p_i*P2+p_j]);
            recv_count += output_dim.size_x[0] * output_dim.size_y[p_i] * output_dim.size_z[p_j];
            
            //start non-blocking send for input data
            MPI_Isend(&send_ptr[send_count], input_dim.size_x[p_i]*input_dim.size_y[p_j]*Nz*sizeof(R_t), 
            MPI_BYTE, p_i*P2+p_j, world_size, MPI_COMM_WORLD, &send_req[p_i*P2+p_j]);
            
            send_count += input_dim.size_x[p_i] * input_dim.size_y[p_j] * Nz;
        }
    }

    MPI_Waitall(world_size-1, send_req.data(), MPI_STATUSES_IGNORE);
    MPI_Barrier(MPI_COMM_WORLD);

    // compute full fft locally
    size_t ws_r2c;

    cufftHandle planR2C;
    cublasHandle_t handle;

    R_t *real    = cuFFT<double>::real(in_d);
    C_t *complex = cuFFT<double>::complex(out_d);

    CUFFT_CALL(cufftCreate(&planR2C));
    CUFFT_CALL(cufftMakePlan3d(planR2C, Nx, Ny, Nz, cuFFT<double>::R2Ctype, &ws_r2c));

    CUFFT_CALL(cuFFT<double>::execR2C(planR2C, real, complex));
    CUDA_CALL(cudaDeviceSynchronize());

    CUBLAS_CALL(cublasCreate(&handle));

    int p;
    do {
        // recv_req contains one null handle (i.e. recv_req[pidx_i]) and P1-1 active handles
        // If all active handles are processed, Waitany will return MPI_UNDEFINED
        MPI_Waitany(world_size-1, recv_req.data(), &p, MPI_STATUSES_IGNORE);
        if (p == MPI_UNDEFINED)
            break;
        
        size_t p_i = p / P2;
        size_t p_j = p % P2;

        cudaMemcpy3DParms cpy_params = {0};
        cpy_params.srcPos = make_cudaPos(0, 0, 0);
        cpy_params.srcPtr = make_cudaPitchedPtr(&recv_ptr[recv_counts[p]], output_dim.size_z[p_j]*sizeof(C_t), output_dim.size_z[p_j], output_dim.size_y[p_i]);
        cpy_params.dstPos = make_cudaPos(output_dim.start_z[p_j]*sizeof(C_t), output_dim.start_y[p_i], 0);
        cpy_params.dstPtr = make_cudaPitchedPtr(res_d, (Nz/2+1)*sizeof(C_t), Nz/2+1, Ny);    
        cpy_params.extent = make_cudaExtent(output_dim.size_z[p_j]*sizeof(C_t), output_dim.size_y[p_i], output_dim.size_x[0]);
        cpy_params.kind   = CUDA_AWARE==1 ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice;   
        
        CUDA_CALL(cudaMemcpy3DAsync(&cpy_params));
    } while (p != MPI_UNDEFINED);
    CUDA_CALL(cudaDeviceSynchronize());
    
    difference<<<(Nx*Ny*(Nz/2+1))/1024+1, 1024>>>(complex, res_d, Nx*Ny*(Nz/2+1));

    double sum = 0;
    CUBLAS_CALL(cublasDzasum(handle, Nx*Ny*(Nz/2+1), complex, 1, &sum));
    CUBLAS_CALL(cublasDestroy(handle));

    printf("\nResults: %f", sum);

    CUFFT_CALL(cufftDestroy(planR2C));

    CUDA_CALL(cudaFree(in_d));
    CUDA_CALL(cudaFree(out_d));
    CUDA_CALL(cudaFree(res_d));
    if (CUDA_AWARE == 1){
        CUDA_CALL(cudaFree(send_ptr));
        CUDA_CALL(cudaFree(recv_ptr));
    } else {
        CUDA_CALL(cudaFreeHost(send_ptr));
        CUDA_CALL(cudaFreeHost(recv_ptr));
    }

    return 0;    
}

int compute(int rank, int distributor){
    MPI_Request send_req;
    MPI_Request recv_req;

    R_t *in_d, *recv_ptr;
    C_t *out_d, *send_ptr;

    size_t pidx_i = rank / P2;
    size_t pidx_j = rank % P2;

    //initialize MPIcuFFT
    printf("\nworld_size %d\n", distributor + (PartOfCluster==1 ? 1 : 0));
    MPIcuFFT_Pencil<double> mpicuFFT(MPI_COMM_WORLD, CUDA_AWARE==1, distributor + (PartOfCluster==1 ? 1 : 0));

    Pencil_Partition partition(P1, P2);
    GlobalSize global_size(Nx, Ny, Nz);
    mpicuFFT.initFFT(&global_size, &partition, true);

    // Allocate Memory
    Partition_Dimensions input_dim, transposed_dim, output_dim;
    mpicuFFT.getPartitionDimensions(input_dim, transposed_dim, output_dim);

    size_t out_size = std::max(input_dim.size_x[pidx_i]*input_dim.size_y[pidx_j]*(Nz/2+1), transposed_dim.size_x[pidx_i]*transposed_dim.size_y[0]*transposed_dim.size_z[pidx_j]);
    out_size = std::max(out_size, output_dim.size_x[0]*output_dim.size_y[pidx_i]*output_dim.size_z[pidx_j]);

    printf("(%d, %d) out_size: %d", pidx_i, pidx_j, out_size);

    //allocate memory (device)
    CUDA_CALL(cudaMalloc((void **)&in_d, input_dim.size_x[pidx_i]*input_dim.size_y[pidx_j]*Nz*sizeof(R_t)));
    CUDA_CALL(cudaMalloc((void **)&out_d, out_size*sizeof(C_t)));
    
    if (CUDA_AWARE == 1){
        recv_ptr = in_d;
        send_ptr = out_d;
    } else {
        CUDA_CALL(cudaMallocHost((void **)&recv_ptr, input_dim.size_x[pidx_i]*input_dim.size_y[pidx_j]*Nz*sizeof(R_t)));
        CUDA_CALL(cudaMallocHost((void **)&send_ptr, output_dim.size_x[0]*output_dim.size_y[pidx_i]*output_dim.size_z[pidx_j]*sizeof(C_t)));
    }

    //receive input data via MPI
    MPI_Irecv(recv_ptr, input_dim.size_x[pidx_i]*input_dim.size_y[pidx_j]*Nz*sizeof(R_t), MPI_BYTE, distributor, distributor+1, MPI_COMM_WORLD, &recv_req);
    MPI_Wait(&recv_req, MPI_STATUSES_IGNORE);

    if (CUDA_AWARE == 0){
        CUDA_CALL(cudaMemcpyAsync(in_d, recv_ptr, input_dim.size_x[pidx_i]*input_dim.size_y[pidx_j]*Nz*sizeof(R_t), cudaMemcpyHostToDevice));
    }

    MPI_Barrier(MPI_COMM_WORLD);
    CUDA_CALL(cudaDeviceSynchronize());

    //execute
    mpicuFFT.execR2C(out_d, in_d);

    if (CUDA_AWARE == 0){
        CUDA_CALL(cudaMemcpy(send_ptr, out_d, output_dim.size_x[0]*output_dim.size_y[pidx_i]*output_dim.size_z[pidx_j]*sizeof(C_t), cudaMemcpyDeviceToHost));
    }

    MPI_Isend(send_ptr, output_dim.size_x[0]*output_dim.size_y[pidx_i]*output_dim.size_z[pidx_j]*sizeof(C_t), MPI_BYTE, distributor, distributor+1, MPI_COMM_WORLD, &send_req);
    MPI_Wait(&send_req, MPI_STATUSES_IGNORE);
    
    CUDA_CALL(cudaFree(in_d));
    CUDA_CALL(cudaFree(out_d));

    if (CUDA_AWARE == 0) {
        CUDA_CALL(cudaFreeHost(recv_ptr));
        CUDA_CALL(cudaFreeHost(send_ptr));
    }

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

    if (rank == world_size-1){
        coordinate(world_size);
    } else{
        compute(rank, world_size-1);
    }
    
    //finalize
    MPI_Finalize();

    return 0;
}

