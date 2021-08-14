#include "mpicufft_pencil.hpp"
#include "mpicufft_pencil_opt1.hpp"
#include "tests_pencil_random_3d.hpp"
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <iostream>

#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
    printf("Error %d at %s:%d\n",x,__FILE__,__LINE__);\
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

namespace Difference_Pencil_3D {
    // Difference
    // Definition in tests/pencil/base.cu
    __global__ void differenceFloat(cuFFT<float>::C_t* array1, cuFFT<float>::C_t* array2, int n){
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i < n) {
            array1[i].x -= array2[i].x;
            array1[i].y -= array2[i].y;
        }
    }
    
    __global__ void differenceDouble(cuFFT<double>::C_t* array1, cuFFT<double>::C_t* array2, int n){
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i < n) {
            array1[i].x -= array2[i].x;
            array1[i].y -= array2[i].y;
        }
    }

    __global__ void differenceFloatInv(cuFFT<float>::R_t* array1, cuFFT<float>::R_t* array2, int n, cuFFT<float>::R_t scalar){
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i < n) {
            array1[i] -= scalar * array2[i];
        }
    }
    
    __global__ void differenceDoubleInv(cuFFT<double>::R_t* array1, cuFFT<double>::R_t* array2, int n, cuFFT<double>::R_t scalar){
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i < n) {
            array1[i] -= scalar * array2[i];
        }
    }

    __global__ void derivativeCoefficients(cuFFT<float>::C_t* out, int Nx, int Ny, int Nz, int Nz_offset, int Ny_offset, int N1, int N2){
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < Nx*N2*N1) {
            // get loop variables
            int z = i % N1;
            int y = (i-z)/N1 % N2;
            int x = (i-z-y*N1) / (N1*N2);

            int k1 = 0, k2 = 0, k3 = 0;

            if (x < Nx/2) k1 = x;
            else if (x > (int)(Nx/2)) k1 = Nx - x;

            if (y+Ny_offset < Ny/2) k2 = y+Ny_offset;
            else if (y+Ny_offset > (int)(Ny/2)) k2 = Ny - y - Ny_offset;

            if (z+Nz_offset < Nz/2) k3 = z+Nz_offset;

            double scale = -powf(k1, 2)-powf(k2, 2)-powf(k3, 2);

            out[x*N2*N1+y*N1+z].x = static_cast<float>(static_cast<double>(out[x*N2*N1+y*N1+z].x)*scale/sqrtf(Nx*Ny*Nz));
            out[x*N2*N1+y*N1+z].y = static_cast<float>(static_cast<double>(out[x*N2*N1+y*N1+z].y)*scale/sqrtf(Nx*Ny*Nz));
        }
    }

    __global__ void derivativeCoefficients(cuFFT<double>::C_t* out, int Nx, int Ny, int Nz, int Nz_offset, int Ny_offset, int N1, int N2){
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < Nx*N2*N1) {
            // get loop variables
            int z = i % N1;
            int y = (i-z)/N1 % N2;
            int x = (i-z-y*N1) / (N1*N2);

            int k1 = 0, k2 = 0, k3 = 0;

            if (x < Nx/2) k1 = x;
            else if (x > (int)(Nx/2)) k1 = Nx - x;

            if (y+Ny_offset < Ny/2) k2 = y+Ny_offset;
            else if (y+Ny_offset > (int)(Ny/2)) k2 = Ny - y - Ny_offset;

            if (z+Nz_offset < Nz/2) k3 = z+Nz_offset;

            double scale = -powf(k1, 2)-powf(k2, 2)-powf(k3, 2);

            out[x*N2*N1+y*N1+z].x = out[x*N2*N1+y*N1+z].x*scale/sqrtf(Nx*Ny*Nz);
            out[x*N2*N1+y*N1+z].y = out[x*N2*N1+y*N1+z].y*scale/sqrtf(Nx*Ny*Nz);
        }
    }
    
    template<typename T> 
    struct Difference { 
        static decltype(differenceFloat)* difference;
        static decltype(differenceFloatInv)* differenceInv;
    };
    template<typename T> decltype(differenceFloat)* Difference<T>::difference = differenceFloat;
    template<typename T> decltype(differenceFloatInv)* Difference<T>::differenceInv = differenceFloatInv;
    
    template<> struct Difference<double> { 
        static decltype(differenceDouble)* difference;
        static decltype(differenceDoubleInv)* differenceInv;
    };
    decltype(differenceDouble)* Difference<double>::difference = differenceDouble;
    decltype(differenceDoubleInv)* Difference<double>::differenceInv = differenceDoubleInv;    
}

template<typename T> 
int Tests_Pencil_Random_3D<T>::run(const int testcase, const int opt, const int runs){
    if (testcase == 0)
        return this->testcase0(opt, runs);
    else if (testcase == 1)
        return this->testcase1(opt, runs);
    else if (testcase == 2)
        return this->testcase2(opt, runs);
    else if (testcase == 3)
        return this->testcase3(opt, runs);
    else if (testcase == 4)
        return this->testcase4(opt, runs);
    return -1;
}

template<typename T> 
int Tests_Pencil_Random_3D<T>::testcase0(const int opt, const int runs){
    using R_t = typename cuFFT<T>::R_t;
    using C_t = typename cuFFT<T>::C_t;

    int provided; 
    //initialize MPI
    MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, &provided);
    if (provided < MPI_THREAD_MULTIPLE) {
        printf("ERROR: The MPI library does not have full thread support\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    //number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    //get global rank
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int dev_count;
    CUDA_CALL(cudaGetDeviceCount(&dev_count));
    CUDA_CALL(cudaSetDevice(rank % dev_count));

    size_t pidx_i = rank / P2;
    size_t pidx_j = rank % P2;
        
    //initialize MPIcuFFT
    MPIcuFFT_Pencil<T> *mpicuFFT;
    if (opt == 1)
        mpicuFFT = new MPIcuFFT_Pencil_Opt1<T>(config, MPI_COMM_WORLD, world_size);
    else 
        mpicuFFT = new MPIcuFFT_Pencil<T>(config, MPI_COMM_WORLD, world_size);

    Pencil_Partition partition(P1, P2);
    GlobalSize global_size(Nx, Ny, Nz);
    mpicuFFT->initFFT(&global_size, &partition, true);

    // Allocate Memory
    Partition_Dimensions input_dim, transposed_dim, output_dim;
    mpicuFFT->getPartitionDimensions(input_dim, transposed_dim, output_dim);

    size_t out_size = std::max(input_dim.size_x[pidx_i]*input_dim.size_y[pidx_j]*(Nz/2+1), transposed_dim.size_x[pidx_i]*transposed_dim.size_y[0]*transposed_dim.size_z[pidx_j]);
    out_size = std::max(out_size, output_dim.size_x[0]*output_dim.size_y[pidx_i]*output_dim.size_z[pidx_j]);

    R_t *in_d;
    C_t *out_d;
    // C_t *out_h;

    CUDA_CALL(cudaMalloc((void **)&in_d, input_dim.size_x[pidx_i]*input_dim.size_y[pidx_j]*Nz*sizeof(R_t)));
    CUDA_CALL(cudaMalloc((void **)&out_d, out_size*sizeof(C_t)));
    //allocate memory (host)
    // out_h = (T *)calloc(out_size, sizeof(C_t));

    this->initializeRandArray(in_d, input_dim.size_x[pidx_i]*input_dim.size_y[pidx_j]*Nz);
    for (int i = 0; i < runs; i++) {
        MPI_Barrier(MPI_COMM_WORLD);
        mpicuFFT->execR2C(out_d, in_d);
    }

    // CUDA_CALL(cudaMemcpy(out_h, out_d, out_size*sizeof(C_t), cudaMemcpyDeviceToHost));

    //do stuff with out_h / out_d

    //finalize
    MPI_Finalize();

    CUDA_CALL(cudaFree(in_d));
    CUDA_CALL(cudaFree(out_d));
    // free(out_h);

    return 0;
}

template<typename T>
int Tests_Pencil_Random_3D<T>::testcase1(const int opt, const int runs) {      
    int provided; 
    //initialize MPI
    MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, &provided);
    if (provided < MPI_THREAD_MULTIPLE) {
        printf("ERROR: The MPI library does not have full thread support\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    //number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    //get global rank
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == world_size-1){
        this->coordinate(world_size, runs);
    } else{
        this->compute(rank, world_size-1, opt, runs);
    }
    
    //finalize
    MPI_Finalize();

    return 0;
}

template<typename T>
int Tests_Pencil_Random_3D<T>::coordinate(const int world_size, const int runs) {
    using R_t = typename cuFFT<T>::R_t;
    using C_t = typename cuFFT<T>::C_t;

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
    
    if (config.cuda_aware){
        CUDA_CALL(cudaMalloc((void **)&send_ptr, Nx*Ny*Nz*sizeof(R_t)));
        CUDA_CALL(cudaMalloc((void **)&recv_ptr, Nx*Ny*(Nz/2+1)*sizeof(C_t)));
    } else {
        CUDA_CALL(cudaMallocHost((void **)&send_ptr, Nx*Ny*Nz*sizeof(R_t)));
        CUDA_CALL(cudaMallocHost((void **)&recv_ptr, Nx*Ny*(Nz/2+1)*sizeof(C_t)));
    }

    MPI_Comm temp;
    MPI_Comm_split(MPI_COMM_WORLD, MPI_UNDEFINED, 0, &temp);

    for (int i = 0; i < runs; i++) {
        //random initialization of full Nx*Ny*Nz array
        this->initializeRandArray(in_d, Nx*Ny*Nz);
    
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
                cpy_params.kind   = config.cuda_aware==1 ? cudaMemcpyDeviceToDevice : cudaMemcpyDeviceToHost;
                
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
        
        // compute full fft locally
        size_t ws_r2c;
        
        cufftHandle planR2C;
        cublasHandle_t handle;
        
        R_t *real    = cuFFT<T>::real(in_d);
        C_t *complex = cuFFT<T>::complex(out_d);
        
        CUFFT_CALL(cufftCreate(&planR2C));
        CUFFT_CALL(cufftMakePlan3d(planR2C, Nx, Ny, Nz, cuFFT<T>::R2Ctype, &ws_r2c));
        
        MPI_Barrier(MPI_COMM_WORLD);
        CUFFT_CALL(cuFFT<T>::execR2C(planR2C, real, complex));
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
            cpy_params.kind   = config.cuda_aware? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice;   
            
            CUDA_CALL(cudaMemcpy3DAsync(&cpy_params));
        } while (p != MPI_UNDEFINED);
        CUDA_CALL(cudaDeviceSynchronize());
        
        Difference_Pencil_3D::Difference<T>::difference<<<(Nx*Ny*(Nz/2+1))/1024+1, 1024>>>(complex, res_d, Nx*Ny*(Nz/2+1));
    
        T sum = 0;
        CUBLAS_CALL(Random_Tests<T>::cublasSum(handle, Nx*Ny*(Nz/2+1), complex, 1, &sum));

        printf("\nResults: %f\n", sum);
    
        CUBLAS_CALL(cublasDestroy(handle));
        
        CUFFT_CALL(cufftDestroy(planR2C));
        MPI_Barrier(MPI_COMM_WORLD);
    }

    CUDA_CALL(cudaFree(in_d));
    CUDA_CALL(cudaFree(out_d));
    CUDA_CALL(cudaFree(res_d));
    if (config.cuda_aware){
        CUDA_CALL(cudaFree(send_ptr));
        CUDA_CALL(cudaFree(recv_ptr));
    } else {
        CUDA_CALL(cudaFreeHost(send_ptr));
        CUDA_CALL(cudaFreeHost(recv_ptr));
    }

    return 0;    
}

template<typename T>
int Tests_Pencil_Random_3D<T>::compute(const int rank, const int world_size, const int opt, const int runs){
    using R_t = typename cuFFT<T>::R_t;
    using C_t = typename cuFFT<T>::C_t;

    MPI_Request send_req;
    MPI_Request recv_req;

    R_t *in_d, *recv_ptr;
    C_t *out_d, *send_ptr;

    size_t pidx_i = rank / P2;
    size_t pidx_j = rank % P2;

    //initialize MPIcuFFT
    MPIcuFFT_Pencil<T> *mpicuFFT;
    if (opt == 1)
        mpicuFFT = new MPIcuFFT_Pencil_Opt1<T>(config, MPI_COMM_WORLD, world_size);
    else 
        mpicuFFT = new MPIcuFFT_Pencil<T>(config, MPI_COMM_WORLD, world_size);

    Pencil_Partition partition(P1, P2);
    GlobalSize global_size(Nx, Ny, Nz);
    mpicuFFT->initFFT(&global_size, &partition, true);

    // Allocate Memory
    Partition_Dimensions input_dim, transposed_dim, output_dim;
    mpicuFFT->getPartitionDimensions(input_dim, transposed_dim, output_dim);

    size_t out_size = std::max(input_dim.size_x[pidx_i]*input_dim.size_y[pidx_j]*(Nz/2+1), transposed_dim.size_x[pidx_i]*transposed_dim.size_y[0]*transposed_dim.size_z[pidx_j]);
    out_size = std::max(out_size, output_dim.size_x[0]*output_dim.size_y[pidx_i]*output_dim.size_z[pidx_j]);

    //allocate memory (device)
    CUDA_CALL(cudaMalloc((void **)&in_d, input_dim.size_x[pidx_i]*input_dim.size_y[pidx_j]*Nz*sizeof(R_t)));
    CUDA_CALL(cudaMalloc((void **)&out_d, out_size*sizeof(C_t)));
    
    if (config.cuda_aware){
        recv_ptr = in_d;
        send_ptr = out_d;
    } else {
        CUDA_CALL(cudaMallocHost((void **)&recv_ptr, input_dim.size_x[pidx_i]*input_dim.size_y[pidx_j]*Nz*sizeof(R_t)));
        CUDA_CALL(cudaMallocHost((void **)&send_ptr, output_dim.size_x[0]*output_dim.size_y[pidx_i]*output_dim.size_z[pidx_j]*sizeof(C_t)));
    }

    for (int i = 0; i < runs; i++) {
        //receive input data via MPI
        MPI_Irecv(recv_ptr, input_dim.size_x[pidx_i]*input_dim.size_y[pidx_j]*Nz*sizeof(R_t), MPI_BYTE, world_size, world_size+1, MPI_COMM_WORLD, &recv_req);
        MPI_Wait(&recv_req, MPI_STATUSES_IGNORE);
    
        if (!config.cuda_aware){
            CUDA_CALL(cudaMemcpyAsync(in_d, recv_ptr, input_dim.size_x[pidx_i]*input_dim.size_y[pidx_j]*Nz*sizeof(R_t), cudaMemcpyHostToDevice));
        }
    
        MPI_Barrier(MPI_COMM_WORLD);
        CUDA_CALL(cudaDeviceSynchronize());
    
        //execute
        
        mpicuFFT->execR2C(out_d, in_d);
    
        if (!config.cuda_aware){
            CUDA_CALL(cudaMemcpy(send_ptr, out_d, output_dim.size_x[0]*output_dim.size_y[pidx_i]*output_dim.size_z[pidx_j]*sizeof(C_t), cudaMemcpyDeviceToHost));
        }
    
        MPI_Isend(send_ptr, output_dim.size_x[0]*output_dim.size_y[pidx_i]*output_dim.size_z[pidx_j]*sizeof(C_t), MPI_BYTE, world_size, world_size+1, MPI_COMM_WORLD, &send_req);
        MPI_Wait(&send_req, MPI_STATUSES_IGNORE);
        MPI_Barrier(MPI_COMM_WORLD);
        
    }
    
    CUDA_CALL(cudaFree(in_d));
    CUDA_CALL(cudaFree(out_d));

    if (!config.cuda_aware) {
        CUDA_CALL(cudaFreeHost(recv_ptr));
        CUDA_CALL(cudaFreeHost(send_ptr));
    }

    return 0;
}

template<typename T> 
int Tests_Pencil_Random_3D<T>::testcase2(const int opt, const int runs){
    using R_t = typename cuFFT<T>::R_t;
    using C_t = typename cuFFT<T>::C_t;

    int provided; 
    //initialize MPI
    MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, &provided);
    if (provided < MPI_THREAD_MULTIPLE) {
        printf("ERROR: The MPI library does not have full thread support\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    //number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    //get global rank
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int dev_count;
    CUDA_CALL(cudaGetDeviceCount(&dev_count));
    CUDA_CALL(cudaSetDevice(rank % dev_count));

    size_t pidx_i = rank / P2;
    size_t pidx_j = rank % P2;
        
    //initialize MPIcuFFT
    MPIcuFFT_Pencil<T> *mpicuFFT;
    if (opt == 1)
        mpicuFFT = new MPIcuFFT_Pencil_Opt1<T>(config, MPI_COMM_WORLD, world_size);
    else 
        mpicuFFT = new MPIcuFFT_Pencil<T>(config, MPI_COMM_WORLD, world_size);

    Pencil_Partition partition(P1, P2);
    GlobalSize global_size(Nx, Ny, Nz);
    mpicuFFT->initFFT(&global_size, &partition, true);

    // Allocate Memory
    Partition_Dimensions input_dim, transposed_dim, output_dim;
    mpicuFFT->getPartitionDimensions(input_dim, transposed_dim, output_dim);

    size_t out_size = std::max(input_dim.size_x[pidx_i]*input_dim.size_y[pidx_j]*(Nz/2+1), transposed_dim.size_x[pidx_i]*transposed_dim.size_y[0]*transposed_dim.size_z[pidx_j]);
    out_size = std::max(out_size, output_dim.size_x[0]*output_dim.size_y[pidx_i]*output_dim.size_z[pidx_j]);

    R_t *inv_d;
    C_t *out_d;
    // C_t *out_h;

    CUDA_CALL(cudaMalloc((void **)&inv_d, input_dim.size_x[pidx_i]*input_dim.size_y[pidx_j]*Nz*sizeof(R_t)));
    CUDA_CALL(cudaMalloc((void **)&out_d, out_size*sizeof(C_t)));
    //allocate memory (host)
    // out_h = (T *)calloc(out_size, sizeof(C_t));

    this->initializeRandArray(out_d, output_dim.size_x[0]*output_dim.size_y[pidx_i]*output_dim.size_z[pidx_j]);
    for (int i = 0; i < runs; i++) {
        MPI_Barrier(MPI_COMM_WORLD);
        mpicuFFT->execC2R(inv_d, out_d);
    }

    // CUDA_CALL(cudaMemcpy(out_h, out_d, out_size*sizeof(C_t), cudaMemcpyDeviceToHost));

    //do stuff with out_h / out_d

    //finalize
    MPI_Finalize();

    CUDA_CALL(cudaFree(inv_d));
    CUDA_CALL(cudaFree(out_d));
    // free(out_h);

    return 0;
}

template<typename T> 
int Tests_Pencil_Random_3D<T>::testcase3(const int opt, const int runs){
    using R_t = typename cuFFT<T>::R_t;
    using C_t = typename cuFFT<T>::C_t;

    int provided; 
    cublasHandle_t handle;
    //initialize MPI
    MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, &provided);
    if (provided < MPI_THREAD_MULTIPLE) {
        printf("ERROR: The MPI library does not have full thread support\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    //number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    //get global rank
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int dev_count;
    CUDA_CALL(cudaGetDeviceCount(&dev_count));
    CUDA_CALL(cudaSetDevice(rank % dev_count));

    size_t pidx_i = rank / P2;
    size_t pidx_j = rank % P2;
        
    //initialize MPIcuFFT
    MPIcuFFT_Pencil<T> *mpicuFFT;
    if (opt == 1)
        mpicuFFT = new MPIcuFFT_Pencil_Opt1<T>(config, MPI_COMM_WORLD, world_size);
    else 
        mpicuFFT = new MPIcuFFT_Pencil<T>(config, MPI_COMM_WORLD, world_size);

    Pencil_Partition partition(P1, P2);
    GlobalSize global_size(Nx, Ny, Nz);
    mpicuFFT->initFFT(&global_size, &partition, true);

    // Allocate Memory
    Partition_Dimensions input_dim, transposed_dim, output_dim;
    mpicuFFT->getPartitionDimensions(input_dim, transposed_dim, output_dim);

    size_t out_size = std::max(input_dim.size_x[pidx_i]*input_dim.size_y[pidx_j]*(Nz/2+1), transposed_dim.size_x[pidx_i]*transposed_dim.size_y[0]*transposed_dim.size_z[pidx_j]);
    out_size = std::max(out_size, output_dim.size_x[0]*output_dim.size_y[pidx_i]*output_dim.size_z[pidx_j]);

    R_t *in_d, *inv_d;
    C_t *out_d;
    // C_t *out_h;

    CUDA_CALL(cudaMalloc((void **)&in_d, input_dim.size_x[pidx_i]*input_dim.size_y[pidx_j]*Nz*sizeof(R_t)));
    CUDA_CALL(cudaMalloc((void **)&inv_d, input_dim.size_x[pidx_i]*input_dim.size_y[pidx_j]*Nz*sizeof(R_t)));
    CUDA_CALL(cudaMalloc((void **)&out_d, out_size*sizeof(C_t)));
    CUBLAS_CALL(cublasCreate(&handle));
    //allocate memory (host)
    // out_h = (T *)calloc(out_size, sizeof(C_t));

    for (int i = 0; i < runs; i++) {
        this->initializeRandArray(in_d, input_dim.size_x[pidx_i]*input_dim.size_y[pidx_j]*Nz);
        MPI_Barrier(MPI_COMM_WORLD);
        mpicuFFT->execR2C(out_d, in_d);
        MPI_Barrier(MPI_COMM_WORLD);
        mpicuFFT->execC2R(inv_d, out_d);
        MPI_Barrier(MPI_COMM_WORLD);

        //compare difference
        Difference_Pencil_3D::Difference<T>::differenceInv<<<(input_dim.size_x[pidx_i]*input_dim.size_y[pidx_j]*Nz)/1024+1, 1024>>>(inv_d, in_d, input_dim.size_x[pidx_i]*input_dim.size_y[pidx_j]*Nz, Nx*Ny*Nz);
        T sum = 0, max = 0;
        CUBLAS_CALL(Random_Tests<T>::cublasSumInv(handle, input_dim.size_x[pidx_i]*input_dim.size_y[pidx_j]*Nz, inv_d, 1, &sum));
        int maxIndex;
        CUBLAS_CALL(Random_Tests<T>::cublasMaxIndex(handle, input_dim.size_x[pidx_i]*input_dim.size_y[pidx_j]*Nz, inv_d, 1, &maxIndex));
        CUDA_CALL(cudaMemcpy(&max, inv_d+maxIndex-1, sizeof(T), cudaMemcpyDeviceToHost));
        
        double globalsum = 0;
        double globalmax = 0;
        double sum_d = static_cast<double>(sum);
        double max_d = static_cast<double>(max);
        MPI_Allreduce(&sum_d, &globalsum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&max_d, &globalmax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        if (rank == 0) {
            std::cout << "Result (avg): " << globalsum / (Nx*Ny*Nz) << std::endl;
            std::cout << "Result (max): " << globalmax << std::endl;
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }

    // CUDA_CALL(cudaMemcpy(out_h, out_d, out_size*sizeof(C_t), cudaMemcpyDeviceToHost));

    //do stuff with out_h / out_d

    //finalize
    MPI_Finalize();

    CUDA_CALL(cudaFree(in_d));
    CUDA_CALL(cudaFree(out_d));
    // free(out_h);

    return 0;
}

template<typename T> 
int Tests_Pencil_Random_3D<T>::testcase4(const int opt, const int runs){
    using R_t = typename cuFFT<T>::R_t;
    using C_t = typename cuFFT<T>::C_t;

    int provided; 
    cublasHandle_t handle;
    //initialize MPI
    MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, &provided);
    if (provided < MPI_THREAD_MULTIPLE) {
        printf("ERROR: The MPI library does not have full thread support\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    //number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    //get global rank
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int dev_count;
    CUDA_CALL(cudaGetDeviceCount(&dev_count));
    CUDA_CALL(cudaSetDevice(rank % dev_count));

    size_t pidx_i = rank / P2;
    size_t pidx_j = rank % P2;
        
    //initialize MPIcuFFT
    MPIcuFFT_Pencil<T> *mpicuFFT;
    if (opt == 1)
        mpicuFFT = new MPIcuFFT_Pencil_Opt1<T>(config, MPI_COMM_WORLD, world_size);
    else 
        mpicuFFT = new MPIcuFFT_Pencil<T>(config, MPI_COMM_WORLD, world_size);

    Pencil_Partition partition(P1, P2);
    GlobalSize global_size(Nx, Ny, Nz);
    mpicuFFT->initFFT(&global_size, &partition, true);

    // Allocate Memory
    Partition_Dimensions input_dim, transposed_dim, output_dim;
    mpicuFFT->getPartitionDimensions(input_dim, transposed_dim, output_dim);

    size_t out_size = std::max(input_dim.size_x[pidx_i]*input_dim.size_y[pidx_j]*(Nz/2+1), transposed_dim.size_x[pidx_i]*transposed_dim.size_y[0]*transposed_dim.size_z[pidx_j]);
    out_size = std::max(out_size, output_dim.size_x[0]*output_dim.size_y[pidx_i]*output_dim.size_z[pidx_j]);

    R_t *in_d, *inv_d, *der_d;
    R_t *in_h, *der_h;
    C_t *out_d;
    // C_t *out_h;

    CUDA_CALL(cudaMalloc((void **)&in_d, input_dim.size_x[pidx_i]*input_dim.size_y[pidx_j]*Nz*sizeof(R_t)));
    CUDA_CALL(cudaMallocHost((void **)&in_h, input_dim.size_x[pidx_i]*input_dim.size_y[pidx_j]*Nz*sizeof(R_t)));
    CUDA_CALL(cudaMalloc((void **)&der_d, input_dim.size_x[pidx_i]*input_dim.size_y[pidx_j]*Nz*sizeof(R_t)));
    CUDA_CALL(cudaMallocHost((void **)&der_h, input_dim.size_x[pidx_i]*input_dim.size_y[pidx_j]*Nz*sizeof(R_t)));
    CUDA_CALL(cudaMalloc((void **)&inv_d, input_dim.size_x[pidx_i]*input_dim.size_y[pidx_j]*Nz*sizeof(R_t)));
    CUDA_CALL(cudaMalloc((void **)&out_d, out_size*sizeof(C_t)));

    CUBLAS_CALL(cublasCreate(&handle));

    for (int x = 0; x < input_dim.size_x[pidx_i]; x++) {
        for (int y = 0; y < input_dim.size_y[pidx_j]; y++) {
            for (int z = 0; z < Nz; z++) {
                in_h[x*input_dim.size_y[pidx_j]*Nz+y*Nz+z] = (T)(sin(2.0*M_PI*(input_dim.start_x[pidx_i]+x)/Nx)*sin(2.0*M_PI*(input_dim.start_y[pidx_j]+y)/Ny)*sin(2.0*M_PI*z/Nz));
            }
        }
    }

    for (int x = 0; x < input_dim.size_x[pidx_i]; x++) {
        for (int y = 0; y < input_dim.size_y[pidx_j]; y++) {
            for (int z = 0; z < Nz; z++) {
                der_h[x*input_dim.size_y[pidx_j]*Nz+y*Nz+z] = -3.0*sqrt(Nx*Ny*Nz)*sin(2*M_PI*(input_dim.start_x[pidx_i]+x)/Nx)*sin(2*M_PI*(input_dim.start_y[pidx_j]+y)/Ny)*sin(2*M_PI*z/Nz);
            }
        }
    }

    CUDA_CALL(cudaMemcpyAsync(in_d, in_h, input_dim.size_x[pidx_i]*input_dim.size_y[pidx_j]*Nz*sizeof(R_t), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpyAsync(der_d, der_h, input_dim.size_x[pidx_i]*input_dim.size_y[pidx_j]*Nz*sizeof(R_t), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaDeviceSynchronize());

    for (int i = 0; i < runs; i++) {
        mpicuFFT->execR2C(out_d, in_d);
        Difference_Pencil_3D::derivativeCoefficients<<<(Nx*output_dim.size_y[pidx_i]*output_dim.size_z[pidx_j])/1024+1, 1024>>>
            (out_d, Nx, Ny, Nz, output_dim.start_z[pidx_j], output_dim.start_y[pidx_i], output_dim.size_z[pidx_j], output_dim.size_y[pidx_i]);
        MPI_Barrier(MPI_COMM_WORLD);

        mpicuFFT->execC2R(inv_d, out_d);

        //compare difference
        Difference_Pencil_3D::Difference<T>::differenceInv<<<(input_dim.size_x[pidx_i]*input_dim.size_y[pidx_j]*Nz)/1024+1, 1024>>>
            (inv_d, der_d, input_dim.size_x[pidx_i]*input_dim.size_y[pidx_j]*Nz, 1);
        T sum = 0, max = 0;
        CUBLAS_CALL(Random_Tests<T>::cublasSumInv(handle, input_dim.size_x[pidx_i]*input_dim.size_y[pidx_j]*Nz, inv_d, 1, &sum));
        int maxIndex;
        CUBLAS_CALL(Random_Tests<T>::cublasMaxIndex(handle, input_dim.size_x[pidx_i]*input_dim.size_y[pidx_j]*Nz, inv_d, 1, &maxIndex));
        CUDA_CALL(cudaMemcpy(&max, inv_d+maxIndex-1, sizeof(T), cudaMemcpyDeviceToHost));
        
        double globalsum = 0;
        double globalmax = 0;
        double sum_d = static_cast<double>(sum);
        double max_d = static_cast<double>(max);
        MPI_Allreduce(&sum_d, &globalsum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&max_d, &globalmax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        if (rank == 0) {
            std::cout << "Result (avg): " << globalsum / (Nx*Ny*Nz) << std::endl;
            std::cout << "Result (max): " << globalmax << std::endl;
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }

    // CUDA_CALL(cudaMemcpy(out_h, out_d, out_size*sizeof(C_t), cudaMemcpyDeviceToHost));

    //do stuff with out_h / out_d

    //finalize
    MPI_Finalize();

    CUDA_CALL(cudaFree(in_d));
    CUDA_CALL(cudaFree(out_d));
    // free(out_h);

    return 0;
}

template class Tests_Pencil_Random_3D<float>;
template class Tests_Pencil_Random_3D<double>;

