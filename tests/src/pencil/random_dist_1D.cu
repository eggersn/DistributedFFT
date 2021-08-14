#include "tests_pencil_random_1d.hpp"
#include "mpicufft_pencil.hpp"
#include "mpicufft_pencil_opt1.hpp"
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <iostream>

#define error(e) {                  \
    throw std::runtime_error(e);    \
}

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

namespace Difference_Pencil_1D {
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
int Tests_Pencil_Random_1D<T>::run(const int testcase, const int opt, const int runs){
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
int Tests_Pencil_Random_1D<T>::testcase0(const int opt, const int runs){
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

    size_t out_size = input_dim.size_x[pidx_i]*input_dim.size_y[pidx_j]*(Nz/2+1);

    R_t *in_d;
    C_t *out_d;
    // C_t *out_h;

    CUDA_CALL(cudaMalloc((void **)&in_d, input_dim.size_x[pidx_i]*input_dim.size_y[pidx_j]*Nz*sizeof(R_t)));
    CUDA_CALL(cudaMalloc((void **)&out_d, out_size*sizeof(C_t)));
    //allocate memory (host)
    // out_h = (T *)calloc(out_size, sizeof(C_t));

    for (int i = 0; i < runs; i++) {
        this->initializeRandArray(in_d, input_dim.size_x[pidx_i]*input_dim.size_y[pidx_j]*Nz);
        MPI_Barrier(MPI_COMM_WORLD);
        mpicuFFT->execR2C(out_d, in_d, 1);
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
int Tests_Pencil_Random_1D<T>::testcase1(const int opt, const int runs) {      
    if (opt != 0)
        error("Selected option is not supported for this testcase");
    
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
int Tests_Pencil_Random_1D<T>::coordinate(const int world_size, const int runs) {
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
                cpy_params.kind   = config.cuda_aware ? cudaMemcpyDeviceToDevice : cudaMemcpyDeviceToHost;
                
                CUDA_CALL(cudaMemcpy3DAsync(&cpy_params));
                CUDA_CALL(cudaDeviceSynchronize());
    
                recv_counts.push_back(recv_count);
    
                //start non-blocking receive for distributed results (asynch to local fft computation)
                MPI_Irecv(&recv_ptr[recv_count], (Nz/2+1)*input_dim.size_y[p_j]*input_dim.size_x[p_i]*sizeof(C_t), 
                    MPI_BYTE, p_i*P2+p_j, world_size, MPI_COMM_WORLD, &recv_req[p_i*P2+p_j]);
                recv_count += (Nz/2+1)*input_dim.size_y[p_j]*input_dim.size_x[p_i];
    
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
        
        R_t *real    = cuFFT<T>::real(in_d);
        C_t *complex = cuFFT<T>::complex(out_d);
        
        CUFFT_CALL(cufftCreate(&planR2C));
        long long int n[1] = {static_cast<long long int>(input_dim.size_z[0])};
        
        CUFFT_CALL(cufftMakePlanMany64(planR2C, 1, n, nullptr, 0, 0, nullptr, 0, 0, cuFFT<T>::R2Ctype, static_cast<long long int>(Nx*Ny), &ws_r2c));
        
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
            cpy_params.srcPtr = make_cudaPitchedPtr(&recv_ptr[recv_counts[p]], (Nz/2+1)*sizeof(C_t), Nz/2+1, input_dim.size_y[p_j]);
            cpy_params.dstPos = make_cudaPos(0, input_dim.start_y[p_j], input_dim.start_x[p_i]);
            cpy_params.dstPtr = make_cudaPitchedPtr(res_d, (Nz/2+1)*sizeof(C_t), Nz/2+1, Ny);    
            cpy_params.extent = make_cudaExtent((Nz/2+1)*sizeof(C_t), input_dim.size_y[p_j], input_dim.size_x[p_i]);
            cpy_params.kind   = config.cuda_aware ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice;   
            
            CUDA_CALL(cudaMemcpy3DAsync(&cpy_params));
        } while (p != MPI_UNDEFINED);
        
        CUDA_CALL(cudaDeviceSynchronize());
    
        Difference_Pencil_1D::Difference<T>::difference<<<(Nx*Ny*(Nz/2+1))/1024+1, 1024>>>(complex, res_d, Nx*Ny*(Nz/2+1));
        CUDA_CALL(cudaDeviceSynchronize());
    
        T sum = 0;
        CUBLAS_CALL(Random_Tests<T>::cublasSum(handle, Nx*Ny*(Nz/2+1), complex, 1, &sum));
        CUBLAS_CALL(cublasDestroy(handle));
    
        printf("\nResults: %f", sum);
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
int Tests_Pencil_Random_1D<T>::compute(const int rank, const int world_size, const int opt, const int runs){
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
    mpicuFFT = new MPIcuFFT_Pencil<T>(config, MPI_COMM_WORLD, world_size);

    Pencil_Partition partition(P1, P2);
    GlobalSize global_size(Nx, Ny, Nz);
    mpicuFFT->initFFT(&global_size, &partition, true);

    // Allocate Memory
    Partition_Dimensions input_dim, transposed_dim, output_dim;
    mpicuFFT->getPartitionDimensions(input_dim, transposed_dim, output_dim);

    size_t out_size = input_dim.size_x[pidx_i]*input_dim.size_y[pidx_j]*(Nz/2+1);

    //allocate memory (device)
    CUDA_CALL(cudaMalloc((void **)&in_d, input_dim.size_x[pidx_i]*input_dim.size_y[pidx_j]*Nz*sizeof(R_t)));
    CUDA_CALL(cudaMalloc((void **)&out_d, out_size*sizeof(C_t)));
    
    if (config.cuda_aware){
        recv_ptr = in_d;
        send_ptr = out_d;
    } else {
        CUDA_CALL(cudaMallocHost((void **)&recv_ptr, input_dim.size_x[pidx_i]*input_dim.size_y[pidx_j]*Nz*sizeof(R_t)));
        CUDA_CALL(cudaMallocHost((void **)&send_ptr, input_dim.size_x[pidx_i]*input_dim.size_y[pidx_j]*(Nz/2+1)*sizeof(C_t)));
    }

    for (int i = 0; i < runs; i++) {
        //receive input data via MPI
        MPI_Irecv(recv_ptr, input_dim.size_x[pidx_i]*input_dim.size_y[pidx_j]*Nz*sizeof(R_t), MPI_BYTE, world_size, world_size+1, MPI_COMM_WORLD, &recv_req);
        MPI_Wait(&recv_req, MPI_STATUSES_IGNORE);
    
        if (!config.cuda_aware){
            CUDA_CALL(cudaMemcpyAsync(in_d, recv_ptr, input_dim.size_x[pidx_i]*input_dim.size_y[pidx_j]*Nz*sizeof(R_t), cudaMemcpyHostToDevice));
            CUDA_CALL(cudaDeviceSynchronize());
        }
    
        MPI_Barrier(MPI_COMM_WORLD);
    
        //execute
        mpicuFFT->execR2C(out_d, in_d, 1);
    
        if (!config.cuda_aware){
            CUDA_CALL(cudaMemcpyAsync(send_ptr, out_d, input_dim.size_x[pidx_i]*input_dim.size_y[pidx_j]*(Nz/2+1)*sizeof(C_t), cudaMemcpyDeviceToHost));
            CUDA_CALL(cudaDeviceSynchronize());
        }
    
        MPI_Isend(send_ptr, input_dim.size_x[pidx_i]*input_dim.size_y[pidx_j]*(Nz/2+1)*sizeof(C_t), MPI_BYTE, world_size, world_size+1, MPI_COMM_WORLD, &send_req);
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
int Tests_Pencil_Random_1D<T>::testcase2(const int opt, const int runs){
    return 0;
}

template<typename T> 
int Tests_Pencil_Random_1D<T>::testcase3(const int opt, const int runs){
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

    size_t out_size = input_dim.size_x[pidx_i]*input_dim.size_y[pidx_j]*(Nz/2+1);

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
        mpicuFFT->execR2C(out_d, in_d, 1);
        MPI_Barrier(MPI_COMM_WORLD);
        mpicuFFT->execC2R(inv_d, out_d, 1);
        MPI_Barrier(MPI_COMM_WORLD);

        //compare difference
        Difference_Pencil_1D::Difference<T>::differenceInv<<<(input_dim.size_x[pidx_i]*input_dim.size_y[pidx_j]*Nz)/1024+1, 1024>>>(inv_d, in_d, input_dim.size_x[pidx_i]*input_dim.size_y[pidx_j]*Nz, Nz);
        T sum = 0;
        CUBLAS_CALL(Random_Tests<T>::cublasSumInv(handle, input_dim.size_x[pidx_i]*input_dim.size_y[pidx_j]*Nz, inv_d, 1, &sum));
        
        double globalsum = 0;
        double sum_d = static_cast<double>(sum);
        MPI_Allreduce(&sum_d, &globalsum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        if (rank == 0)
            std::cout << "Result: " << globalsum << std::endl;
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
int Tests_Pencil_Random_1D<T>::testcase4(const int opt, const int runs){
    return 0;
}

template class Tests_Pencil_Random_1D<float>;
template class Tests_Pencil_Random_1D<double>;