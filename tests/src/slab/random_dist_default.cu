#include "mpicufft_slab.hpp"
#include "mpicufft_slab_opt1.hpp"
#include "tests_slab_random_default.hpp"
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

namespace Difference_Slab_Default {
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
    
    template<typename T> 
    struct Difference { 
        static decltype(differenceFloat)* difference;
    };
    template<typename T> decltype(differenceFloat)* Difference<T>::difference = differenceFloat;
    
    template<> struct Difference<double> { 
        static decltype(differenceDouble)* difference;
    };
    decltype(differenceDouble)* Difference<double>::difference = differenceDouble;    
}

template<typename T> 
int Tests_Slab_Random_Default<T>::run(const int testcase, const int opt, const int runs){
    if (testcase == 0)
        return this->testcase0(opt, runs);
    else if (testcase == 1)
        return this->testcase1(opt, runs);
    return -1;
}

template<typename T> 
int Tests_Slab_Random_Default<T>::testcase0(const int opt, const int runs){
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
    size_t N2=Ny/world_size;
    if (rank < Nx%world_size)
        N1++;
    if (rank < Ny%world_size)
        N2++;

    R_t *in_d;
    C_t *out_d;
    // C_t *out_h;
    size_t out_size = std::max(N1*Ny*(Nz/2+1), Nx*N2*(Nz/2+1));

    //allocate memory (device)
    CUDA_CALL(cudaMalloc((void **)&in_d, N1*Ny*Nz*sizeof(R_t)));
    CUDA_CALL(cudaMalloc((void **)&out_d, out_size*sizeof(C_t)));
    //allocate memory (host)
    // out_h = (C_t *)calloc(out_size, sizeof(C_t));
    
    MPIcuFFT_Slab<T> *mpicuFFT;
    if (opt == 1)
        mpicuFFT = new MPIcuFFT_Slab_Opt1<T>(config, MPI_COMM_WORLD, world_size);
    else 
        mpicuFFT = new MPIcuFFT_Slab<T>(config, MPI_COMM_WORLD, world_size);
        

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

    //do stuff

    //finalize
    MPI_Finalize();

    CUDA_CALL(cudaFree(in_d));
    CUDA_CALL(cudaFree(out_d));
    // free(out_h);

    delete mpicuFFT;
    return 0;
}

template<typename T>
int Tests_Slab_Random_Default<T>::testcase1(const int opt, const int runs) {     
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

template<typename T> 
int Tests_Slab_Random_Default<T>::coordinate(const int world_size, const int runs){
    using R_t = typename cuFFT<T>::R_t;
    using C_t = typename cuFFT<T>::C_t;

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
        this->initializeRandArray(in_d, Nx);
    
        //Copy input data to send-buffer and initialize cufft
        CUDA_CALL(cudaMemcpyAsync(send_ptr, in_d, Nx*Ny*Nz*sizeof(R_t), 
            config.cuda_aware?cudaMemcpyDeviceToDevice:cudaMemcpyDeviceToHost));
    
        CUFFT_CALL(cufftCreate(&planR2C));
        CUFFT_CALL(cufftSetAutoAllocation(planR2C, 0));
        CUFFT_CALL(cufftMakePlan3d(planR2C, Nx, Ny, Nz, cuFFT<T>::R2Ctype, &ws_r2c));
        CUFFT_CALL(cufftSetWorkArea(planR2C, in_d));

        MPI_Comm temp;
        MPI_Comm_split(MPI_COMM_WORLD, MPI_UNDEFINED, 0, &temp);
    
        //Distribute input data
        size_t N1 = Nx/world_size;
        size_t N2 = Ny/world_size;
        size_t send_count = 0;
        size_t recv_count = 0;
        std::vector<size_t> recv_counts;
        std::vector<size_t> ostarty;
        ostarty.push_back(0);
        for (int pidx = 0; pidx < world_size; pidx++){
            size_t Nxpidx = N1 + (pidx<Nx%world_size?1:0);
            size_t Nypidx = N2 + (pidx<Ny%world_size?1:0);
            recv_req[pidx] = MPI_REQUEST_NULL;
            send_req[pidx] = MPI_REQUEST_NULL;
            ostarty.push_back(ostarty[pidx]+Nypidx);
    
            //start non-blocking receive for distributed results (asynch to local fft computation)
            MPI_Irecv(&recv_ptr[recv_count], Nx*Nypidx*(Nz/2+1)*sizeof(C_t), MPI_BYTE, pidx, pidx, MPI_COMM_WORLD, &recv_req[pidx]);
            recv_counts.push_back(recv_count);
            recv_count += Nx*Nypidx*(Nz/2+1);
    
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
    
            size_t osizey = N2 + (p<Ny%world_size?1:0);
            
            cudaMemcpy3DParms cpy_params = {0};
            cpy_params.srcPos = make_cudaPos(0, 0, 0);
            cpy_params.srcPtr = make_cudaPitchedPtr(&recv_ptr[recv_counts[p]], (Nz/2+1)*sizeof(C_t), Nz/2+1, osizey);
            cpy_params.dstPos = make_cudaPos(0, ostarty[p], 0);
            cpy_params.dstPtr = make_cudaPitchedPtr(res_d, (Nz/2+1)*sizeof(C_t), Nz/2+1, Ny);    
            cpy_params.extent = make_cudaExtent((Nz/2+1)*sizeof(C_t), osizey, Nx);
            cpy_params.kind   = config.cuda_aware ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice;   
            
            CUDA_CALL(cudaMemcpy3DAsync(&cpy_params));
        } while (p != MPI_UNDEFINED);
        CUDA_CALL(cudaDeviceSynchronize());
    
        //compare difference
        Difference_Slab_Default::Difference<T>::difference<<<(Nx*Ny*(Nz/2+1))/1024+1, 1024>>>(complex, res_d, Nx*Ny*(Nz/2+1));
    
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

template<typename T> 
int Tests_Slab_Random_Default<T>::compute(const int rank, const int world_size, const int opt, const int runs){
    using R_t = typename cuFFT<T>::R_t;
    using C_t = typename cuFFT<T>::C_t;

    std::vector<MPI_Request> send_req;
    std::vector<MPI_Request> recv_req;

    size_t N1=Nx/world_size;
    size_t N2=Ny/world_size;
    if (rank < Nx%world_size)
        N1++;
    if (rank < Ny%world_size)
        N2++;

    R_t *in_d, *recv_ptr;
    C_t *out_d, *send_ptr;
    size_t out_size = std::max(N1*Ny*(Nz/2+1), Nx*N2*(Nz/2+1));

    //allocate memory (device)
    CUDA_CALL(cudaMalloc((void **)&in_d, N1*Ny*Nz*sizeof(R_t)));
    CUDA_CALL(cudaMalloc((void **)&out_d, out_size*sizeof(C_t)));
    
    if (config.cuda_aware){
        recv_ptr = in_d;
        send_ptr = out_d;
    } else {
        CUDA_CALL(cudaMallocHost((void **)&recv_ptr, N1*Ny*Nz*sizeof(R_t)));
        CUDA_CALL(cudaMallocHost((void **)&send_ptr, Nx*N2*(Nz/2+1)*sizeof(C_t)));
    }

    MPIcuFFT_Slab<T> *mpicuFFT;
    if (opt == 1)
        mpicuFFT = new MPIcuFFT_Slab_Opt1<T>(config, MPI_COMM_WORLD, world_size);
    else 
        mpicuFFT = new MPIcuFFT_Slab<T>(config, MPI_COMM_WORLD, world_size);
        
    GlobalSize global_size(Nx, Ny, Nz);
    mpicuFFT->initFFT(&global_size, true);

    for (int i = 0; i < runs; i++) {
        send_req.resize(1, MPI_REQUEST_NULL);
        recv_req.resize(1, MPI_REQUEST_NULL);
    
        //receive input data via MPI
        MPI_Irecv(recv_ptr, N1*Ny*Nz*sizeof(R_t), MPI_BYTE, world_size, rank, MPI_COMM_WORLD, &recv_req[0]);
        MPI_Wait(&recv_req[0], MPI_STATUSES_IGNORE);
    
        if (!config.cuda_aware){
            CUDA_CALL(cudaMemcpyAsync(in_d, recv_ptr, N1*Ny*Nz*sizeof(R_t), cudaMemcpyHostToDevice));
            CUDA_CALL(cudaDeviceSynchronize());
        }

        MPI_Barrier(MPI_COMM_WORLD);
        //execute
        mpicuFFT->execR2C(out_d, in_d);

        if (!config.cuda_aware){
            CUDA_CALL(cudaMemcpyAsync(send_ptr, out_d, Nx*N2*(Nz/2+1)*sizeof(C_t), cudaMemcpyDeviceToHost));
            CUDA_CALL(cudaDeviceSynchronize());
        }
    
        MPI_Isend(send_ptr, Nx*N2*(Nz/2+1)*sizeof(C_t), MPI_BYTE, world_size, rank, MPI_COMM_WORLD, &send_req[0]);
        MPI_Wait(&send_req[0], MPI_STATUSES_IGNORE);
        MPI_Barrier(MPI_COMM_WORLD);
    }
    
    CUDA_CALL(cudaFree(in_d));
    CUDA_CALL(cudaFree(out_d));

    delete mpicuFFT;
    return 0;
}

template class Tests_Slab_Random_Default<float>;
template class Tests_Slab_Random_Default<double>;
