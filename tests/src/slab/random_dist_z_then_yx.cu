#include "mpicufft_slab_z_then_yx.hpp"
#include "mpicufft_slab_z_then_yx_opt1.hpp"
#include "tests_slab_random_z_then_yx.hpp"
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

namespace Difference_Slab_Z_Then_YX {
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
        static decltype(differenceDoubleInv)* differenceInv;
        static decltype(differenceDouble)* difference;
    };
    decltype(differenceDouble)* Difference<double>::difference = differenceDouble;    
    decltype(differenceDoubleInv)* Difference<double>::differenceInv = differenceDoubleInv;   
}

template<typename T> 
int Tests_Slab_Random_Z_Then_YX<T>::run(const int testcase, const int opt, const int runs){
    if (testcase == 0)
        return this->testcase0(opt, runs);
    else if (testcase == 1)
        return this->testcase1(opt, runs);
    else if (testcase == 2)
        return this->testcase2(opt, runs);
    return -1;
}

template<typename T> 
int Tests_Slab_Random_Z_Then_YX<T>::testcase0(const int opt, const int runs){
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

    size_t N1=Nx/world_size;
    size_t N2=(Nz/2+1)/world_size;
    if (rank < Nx%world_size)
        N1++;
    if (rank < (Nz/2+1)%world_size)
        N2++;

    R_t *in_d;
    C_t *out_d;
    size_t out_size = std::max(N1*Ny*(Nz/2+1), Nx*Ny*N2);

    //allocate memory (device)
    CUDA_CALL(cudaMalloc((void **)&in_d, N1*Ny*Nz*sizeof(R_t)));
    CUDA_CALL(cudaMalloc((void **)&out_d, out_size*sizeof(C_t)));
    //allocate memory (host)
    // out_h = (C_t *)calloc(out_size, sizeof(C_t));
    
    MPIcuFFT_Slab_Z_Then_YX<T> *mpicuFFT;
    if (opt == 1)
        mpicuFFT = new MPIcuFFT_Slab_Z_Then_YX_Opt1<T>(config, MPI_COMM_WORLD, world_size);
    else 
        mpicuFFT = new MPIcuFFT_Slab_Z_Then_YX<T>(config, MPI_COMM_WORLD, world_size);
    
    GlobalSize global_size(Nx, Ny, Nz);
    mpicuFFT->initFFT(&global_size, true);
    //execute
    for (int i = 0; i < runs; i++){
        this->initializeRandArray(in_d, N1);
        MPI_Barrier(MPI_COMM_WORLD);
        mpicuFFT->execR2C(out_d, in_d);
        MPI_Barrier(MPI_COMM_WORLD);
    }
    
    // CUDA_CALL(cudaMemcpy(out_h, out_d, out_size*sizeof(C_t), cudaMemcpyDeviceToHost));

    //do stuff with out_h

    //finalize
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();

    CUDA_CALL(cudaFree(in_d));
    CUDA_CALL(cudaFree(out_d));
    // free(out_h);

    delete mpicuFFT;
    return 0;
}

template<typename T>
int Tests_Slab_Random_Z_Then_YX<T>::testcase1(const int opt, const int runs) {      
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
    world_size--;

    //get global rank
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == world_size){
        this->coordinate(world_size, runs);
    } else{
        this->compute(rank, world_size, opt, runs);
    }
    
    //finalize
    MPI_Finalize();

    return 0;
}

template <typename T>
int Tests_Slab_Random_Z_Then_YX<T>::coordinate(const int world_size, const int runs){
    using R_t = typename cuFFT<T>::R_t;
    using C_t = typename cuFFT<T>::C_t;

    std::vector<MPI_Request> send_req;
    std::vector<MPI_Request> recv_req;

    R_t *in_d, *send_ptr;
    C_t *out_d, *recv_ptr, *res_d;

    size_t ws_r2c;

    cufftHandle planR2C;
    cublasHandle_t handle;

    //allocate memory (device)
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
        send_req.resize(world_size, MPI_REQUEST_NULL);
        recv_req.resize(world_size, MPI_REQUEST_NULL);
        //random initialization of full Nx*Ny*Nz array
        this->initializeRandArray(in_d, Nx);
    
        //Copy input data to send-buffer and initialize cufft
        CUDA_CALL(cudaMemcpyAsync(send_ptr, in_d, Nx*Ny*Nz*sizeof(R_t), 
            config.cuda_aware ? cudaMemcpyDeviceToDevice:cudaMemcpyDeviceToHost));
    
        CUFFT_CALL(cufftCreate(&planR2C));
        CUFFT_CALL(cufftSetAutoAllocation(planR2C, 0));
        CUFFT_CALL(cufftMakePlan3d(planR2C, Nx, Ny, Nz, cuFFT<T>::R2Ctype, &ws_r2c));
        CUFFT_CALL(cufftSetWorkArea(planR2C, in_d));
    
        //Distribute input data
        size_t N1 = Nx/world_size;
        size_t N2 = (Nz/2+1)/world_size;
        size_t send_count = 0;
        size_t recv_count = 0;
        std::vector<size_t> recv_counts;
        std::vector<size_t> ostartz;
        ostartz.push_back(0);
        for (int pidx = 0; pidx < world_size; pidx++){
            size_t Nxpidx = N1 + (pidx<Nx%world_size?1:0);
            size_t Nzpidx = N2 + (pidx<(Nz/2+1)%world_size?1:0);
            recv_req[pidx] = MPI_REQUEST_NULL;
            send_req[pidx] = MPI_REQUEST_NULL;
            ostartz.push_back(ostartz[pidx]+Nzpidx);
    
            //start non-blocking receive for distributed results (asynch to local fft computation)
            MPI_Irecv(&recv_ptr[recv_count], Nx*Ny*Nzpidx*sizeof(C_t), MPI_BYTE, pidx, pidx, MPI_COMM_WORLD, &recv_req[pidx]);
            recv_counts.push_back(recv_count);
            recv_count += Nx*Ny*Nzpidx;
    
            //start non-blocking send for input data
            MPI_Isend(&send_ptr[send_count], Nxpidx*Ny*Nz*sizeof(R_t), MPI_BYTE, pidx, pidx, MPI_COMM_WORLD, &send_req[pidx]);
            send_count += Nxpidx*Ny*Nz;
        }
    
        //wait till all input data has been distributed
        MPI_Waitall(world_size, send_req.data(), MPI_STATUSES_IGNORE);
    
        //compute local fft
        R_t *real    = cuFFT<T>::real(in_d);
        C_t *complex = cuFFT<T>::complex(out_d);
        MPI_Barrier(MPI_COMM_WORLD);
        CUFFT_CALL(cuFFT<T>::execR2C(planR2C, real, complex));
        CUDA_CALL(cudaDeviceSynchronize());
    
        CUBLAS_CALL(cublasCreate(&handle));
    
        int p;
        do {
            // recv_req contains one null handle (i.e. recv_req[pidx_i]) and P1-1 active handles
            // If all active handles are processed, Waitany will return MPI_UNDEFINED
            MPI_Waitany(world_size, recv_req.data(), &p, MPI_STATUSES_IGNORE);
            if (p == MPI_UNDEFINED)
                break;
    
            size_t osizez = N2 + (p<(Nz/2+1)%world_size?1:0);
            
            cudaMemcpy3DParms cpy_params = {0};
            cpy_params.srcPos = make_cudaPos(0, 0, 0);
            cpy_params.srcPtr = make_cudaPitchedPtr(&recv_ptr[recv_counts[p]], osizez*sizeof(C_t), osizez, Ny);
            cpy_params.dstPos = make_cudaPos(ostartz[p]*sizeof(C_t), 0, 0);
            cpy_params.dstPtr = make_cudaPitchedPtr(res_d, (Nz/2+1)*sizeof(C_t), Nz/2+1, Ny);    
            cpy_params.extent = make_cudaExtent(osizez*sizeof(C_t), Ny, Nx);
            cpy_params.kind   = config.cuda_aware ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice;   
            
            CUDA_CALL(cudaMemcpy3DAsync(&cpy_params));
        } while (p != MPI_UNDEFINED);
        CUDA_CALL(cudaDeviceSynchronize());
    
        //compare difference
        Difference_Slab_Z_Then_YX::Difference<T>::difference<<<(Nx*Ny*(Nz/2+1))/1024+1, 1024>>>(complex, res_d, Nx*Ny*(Nz/2+1));
    
        T sum = 0;
        CUBLAS_CALL(Random_Tests<T>::cublasSum(handle, Nx*Ny*(Nz/2+1), complex, 1, &sum));
        CUBLAS_CALL(cublasDestroy(handle));
    
        std::cout << "Result " << sum << std::endl;
    
        CUFFT_CALL(cufftDestroy(planR2C));
        MPI_Barrier(MPI_COMM_WORLD);
    }

    CUDA_CALL(cudaFree(in_d));
    CUDA_CALL(cudaFree(out_d));
    if (!config.cuda_aware){
        CUDA_CALL(cudaFree(res_d));
    } 

    return 0;
}

template <typename T>
int Tests_Slab_Random_Z_Then_YX<T>::compute(const int rank, const int world_size, const int opt, const int runs){
    using R_t = typename cuFFT<T>::R_t;
    using C_t = typename cuFFT<T>::C_t;

    MPI_Request send_req;
    MPI_Request recv_req;

    size_t N1=Nx/world_size;
    size_t N2=(Nz/2+1)/world_size;
    if (rank < Nx%world_size)
        N1++;
    if (rank < (Nz/2+1)%world_size)
        N2++;

    R_t *in_d, *recv_ptr;
    C_t *out_d, *send_ptr;
    size_t out_size = std::max(N1*Ny*(Nz/2+1), Nx*Ny*N2);

    //allocate memory (device)
    CUDA_CALL(cudaMalloc((void **)&in_d, N1*Ny*Nz*sizeof(R_t)));
    CUDA_CALL(cudaMalloc((void **)&out_d, out_size*sizeof(C_t)));
    
    if (config.cuda_aware){
        recv_ptr = in_d;
        send_ptr = out_d;
    } else {
        CUDA_CALL(cudaMallocHost((void **)&recv_ptr, N1*Ny*Nz*sizeof(R_t)));
        CUDA_CALL(cudaMallocHost((void **)&send_ptr, Nx*Ny*N2*sizeof(C_t)));
    }

    MPIcuFFT_Slab_Z_Then_YX<T> *mpicuFFT;
    if (opt == 1) 
        mpicuFFT = new MPIcuFFT_Slab_Z_Then_YX_Opt1<T>(config, MPI_COMM_WORLD, world_size);
    else 
        mpicuFFT = new MPIcuFFT_Slab_Z_Then_YX<T>(config, MPI_COMM_WORLD, world_size);
    
    GlobalSize global_size(Nx, Ny, Nz);
    mpicuFFT->initFFT(&global_size, true);

    for (int i = 0; i < runs; i++) {
        //receive input data via MPI
        MPI_Irecv(recv_ptr, N1*Ny*Nz*sizeof(R_t), MPI_BYTE, world_size, rank, MPI_COMM_WORLD, &recv_req);
        MPI_Wait(&recv_req, MPI_STATUSES_IGNORE);
    
        if (!config.cuda_aware){
            CUDA_CALL(cudaMemcpyAsync(in_d, recv_ptr, N1*Ny*Nz*sizeof(R_t), cudaMemcpyHostToDevice));
            CUDA_CALL(cudaDeviceSynchronize());
        }

        MPI_Barrier(MPI_COMM_WORLD);
        //execute
        mpicuFFT->execR2C(out_d, in_d);

        if (!config.cuda_aware){
            CUDA_CALL(cudaMemcpyAsync(send_ptr, out_d, Nx*Ny*N2*sizeof(C_t), cudaMemcpyDeviceToHost));
            CUDA_CALL(cudaDeviceSynchronize());
        }
    
        MPI_Isend(send_ptr, Nx*Ny*N2*sizeof(C_t), MPI_BYTE, world_size, rank, MPI_COMM_WORLD, &send_req);
        MPI_Wait(&send_req, MPI_STATUSES_IGNORE);
        MPI_Barrier(MPI_COMM_WORLD);
    }
    
    CUDA_CALL(cudaFree(in_d));
    CUDA_CALL(cudaFree(out_d));

    delete mpicuFFT;
    return 0;
}

template<typename T> 
int Tests_Slab_Random_Z_Then_YX<T>::testcase2(const int opt, const int runs){
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

    size_t N1=Nx/world_size;
    size_t N2=(Nz/2+1)/world_size;
    if (rank < Nx%world_size)
        N1++;
    if (rank < (Nz/2+1)%world_size)
        N2++;

    R_t *in_d, *inv_d;
    R_t *in_h, *inv_h;
    C_t *out_d;

    size_t out_size = std::max(N1*Ny*(Nz/2+1), Nx*Ny*N2);

    //allocate memory (device)
    CUDA_CALL(cudaMalloc((void **)&in_d, N1*Ny*Nz*sizeof(R_t)));
    CUDA_CALL(cudaMalloc((void **)&inv_d, N1*Ny*Nz*sizeof(R_t)));
    CUDA_CALL(cudaMallocHost((void **)&in_h, N1*Ny*Nz*sizeof(R_t)));
    CUDA_CALL(cudaMallocHost((void **)&inv_h, N1*Ny*Nz*sizeof(R_t)));
    CUDA_CALL(cudaMalloc((void **)&out_d, out_size*sizeof(C_t)));
    
    MPIcuFFT_Slab_Z_Then_YX<T> *mpicuFFT;
    if (opt == 1) 
        mpicuFFT = new MPIcuFFT_Slab_Z_Then_YX_Opt1<T>(config, MPI_COMM_WORLD, world_size);
    else 
        mpicuFFT = new MPIcuFFT_Slab_Z_Then_YX<T>(config, MPI_COMM_WORLD, world_size);
        

    GlobalSize global_size(Nx, Ny, Nz);
    mpicuFFT->initFFT(&global_size, true);
    CUBLAS_CALL(cublasCreate(&handle));
    
    //execute
    for (int i = 0; i < runs; i++){
        this->initializeRandArray(in_d, N1);
        MPI_Barrier(MPI_COMM_WORLD);
        mpicuFFT->execR2C(out_d, in_d);
        MPI_Barrier(MPI_COMM_WORLD);
        mpicuFFT->execC2R(inv_d, out_d);
        MPI_Barrier(MPI_COMM_WORLD);
        //compare difference
        Difference_Slab_Z_Then_YX::Difference<T>::differenceInv<<<(N1*Ny*Nz)/1024+1, 1024>>>(inv_d, in_d, N1*Ny*Nz, Nx*Ny*Nz);
        T sum = 0;
        CUBLAS_CALL(Random_Tests<T>::cublasSumInv(handle, N1*Ny*Nz, inv_d, 1, &sum));
        
        double globalsum = 0;
        double sum_d = static_cast<double>(sum);
        MPI_Allreduce(&sum_d, &globalsum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        if (rank == 0)
            std::cout << "Result: " << globalsum << std::endl;
            
    }
    
    CUBLAS_CALL(cublasDestroy(handle));
    //finalize
    MPI_Finalize();

    CUDA_CALL(cudaFree(in_d));
    CUDA_CALL(cudaFree(inv_d));
    CUDA_CALL(cudaFree(out_d));

    delete mpicuFFT;
    return 0;
}

template class Tests_Slab_Random_Z_Then_YX<float>;
template class Tests_Slab_Random_Z_Then_YX<double>;


