#include "tests_reference.hpp"
#include "params.hpp"
#include "cufft.hpp"
#include <iostream>
#include <cufft.h>
#include <vector>
#include <thread> 
#include <mutex>
#include <condition_variable>

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

template<typename T>
int Tests_Reference<T>::initializeRandArray(void* in_d, size_t N1, size_t N2){
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

template<typename T>
int Tests_Reference<T>::run(const int testcase, const int opt, const int runs) {
    if (Nx == 0)
        throw std::runtime_error("Not initialized");
    if (testcase == 0)
        return this->testcase0(runs);
    else if (testcase == 1)
        return this->testcase1(opt, runs);
    else if (testcase == 2)
        return this->testcase2(opt, runs);
    throw std::runtime_error("Invalid Testcase!");
}

template<typename T>
int Tests_Reference<T>::testcase0(const int runs) {
    using R_t = typename cuFFT<T>::R_t;
    using C_t = typename cuFFT<T>::C_t;

    MPI_Init(NULL, NULL);

    //number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    //get global rank
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    timer = new Timer(MPI_COMM_WORLD, 0, world_size, rank, section_descriptions, "../benchmarks/reference.csv");
    timer->start();

    size_t pidx_i = rank / P2;
    size_t pidx_j = rank % P2;

    R_t *in_d;
    C_t *out_d;

    Partition_Dimensions dim;

    // Determine all Partition_Dimensions
    // dim:
    dim.size_x.resize(P1, Nx/P1);
    for (int i = 0; i < Nx%P1; i++)
        dim.size_x[i]++;
    dim.size_y.resize(P2, Ny/P2);
    for (int j = 0; j < Ny%P2; j++)
        dim.size_y[j]++;
    dim.size_z.resize(1, Nz);
    dim.computeOffsets();
    
    // Generate random input data for each partition
    // Allocate memory (device)
    CUDA_CALL(cudaMalloc((void **)&in_d, dim.size_x[pidx_i]*dim.size_y[pidx_j]*Nz*sizeof(R_t)));
    CUDA_CALL(cudaMalloc((void **)&out_d, dim.size_x[pidx_i]*dim.size_y[pidx_j]*(Nz/2+1)*sizeof(C_t)));
    
    if (rank == 0) {
        R_t *recv_ptr;
        C_t *send_ptr;

        std::vector<MPI_Request> send_req(world_size, MPI_REQUEST_NULL);
        std::vector<MPI_Request> recv_req(world_size, MPI_REQUEST_NULL);

        R_t *real;
        C_t *complex;
        size_t ws_r2c;
        cufftHandle planR2C;

        CUDA_CALL(cudaMalloc((void **)&real, Nx*Ny*Nz*sizeof(R_t)));
        CUDA_CALL(cudaMalloc((void **)&complex, Nx*Ny*(Nz/2+1)*sizeof(C_t)));

        MPI_Barrier(MPI_COMM_WORLD);

        CUDA_CALL(cudaMallocHost((void **)&recv_ptr, Nx*Ny*Nz*sizeof(R_t)));
        CUDA_CALL(cudaMallocHost((void **)&send_ptr, Nx*Ny*(Nz/2+1)*sizeof(C_t)));

        CUFFT_CALL(cufftCreate(&planR2C));
        CUFFT_CALL(cufftMakePlan3d(planR2C, Nx, Ny, Nz, cuFFT<T>::R2Ctype, &ws_r2c));
        CUDA_CALL(cudaDeviceSynchronize());
        timer->stop_store("init");
        for (int i = 0; i < runs; i++) {
            this->initializeRandArray(in_d, dim.size_x[pidx_i], dim.size_y[pidx_j]);
            // sync before each run
            MPI_Barrier(MPI_COMM_WORLD);
            timer->start();
            // rank 0 has its input data currently stored at in_d.
            CUDA_CALL(cudaMemcpy2DAsync(real, Nz*Ny*sizeof(R_t), in_d, Nz*dim.size_y[0]*sizeof(R_t), Nz*dim.size_y[0]*sizeof(R_t), dim.size_x[0], cudaMemcpyDeviceToDevice));
            CUDA_CALL(cudaDeviceSynchronize());
    
            // receive the input data from the other ranks
            std::vector<size_t> recv_counts;
            size_t recv_count = 0;
            for (int p_i = 0; p_i < P1; p_i++) {
                for (int p_j = 0; p_j < P2; p_j++) {
                    recv_counts.push_back(recv_count);
                    if (p_i + p_j == 0) {
                        recv_count += dim.size_x[p_i]*dim.size_y[p_j]*Nz;
                    } else {
                        MPI_Irecv(&recv_ptr[recv_count], dim.size_x[p_i]*dim.size_y[p_j]*Nz*sizeof(R_t),
                            MPI_BYTE, p_i*P2+p_j, 0, MPI_COMM_WORLD, &recv_req[p_i*P2+p_j]);
                        recv_count += dim.size_x[p_i]*dim.size_y[p_j]*Nz;
                    }               
    
                }
            }
    
            int p;
            do {
                // recv_req contains one null handle (i.e. recv_req[pidx_i]) and P1-1 active handles
                // If all active handles are processed, Waitany will return MPI_UNDEFINED
                MPI_Waitany(world_size, recv_req.data(), &p, MPI_STATUSES_IGNORE);
    
                if (p == MPI_UNDEFINED)
                    break;
                
                size_t p_i = p / P2;
                size_t p_j = p % P2;
        
                cudaMemcpy3DParms cpy_params = {0};
                cpy_params.srcPos = make_cudaPos(0, 0, 0);
                cpy_params.srcPtr = make_cudaPitchedPtr(&recv_ptr[recv_counts[p]], Nz*sizeof(R_t), Nz, dim.size_y[p_j]);
                cpy_params.dstPos = make_cudaPos(0, dim.start_y[p_j], dim.start_x[p_i]);
                cpy_params.dstPtr = make_cudaPitchedPtr(real, Nz*sizeof(R_t), Nz, Ny);    
                cpy_params.extent = make_cudaExtent(Nz*sizeof(R_t), dim.size_y[p_j], dim.size_x[p_i]);
                cpy_params.kind   = cudaMemcpyHostToDevice;   
                
                CUDA_CALL(cudaMemcpy3DAsync(&cpy_params));
            } while (p != MPI_UNDEFINED);
            CUDA_CALL(cudaDeviceSynchronize());
            timer->stop_store("Finished Receive");
    
            // compute the global 3D FFT
            CUFFT_CALL(cuFFT<T>::execR2C(planR2C, real, complex));
            CUDA_CALL(cudaDeviceSynchronize());
            timer->stop_store("3D FFT");

            // redistribute the computed result
            size_t send_count = 0;
            for (int p_i = 0; p_i < P1; p_i++) {
                for (int p_j = 0; p_j < P2; p_j++) {
                    if (p_i + p_j == 0) {
                        // local copy
                        cudaMemcpy3DParms cpy_params = {0};
                        cpy_params.srcPos = make_cudaPos(0, dim.start_y[p_j], dim.start_x[p_i]);
                        cpy_params.srcPtr = make_cudaPitchedPtr(complex, (Nz/2+1)*sizeof(C_t), Nz/2+1, Ny);
                        cpy_params.dstPos = make_cudaPos(0, 0, 0);
                        cpy_params.dstPtr = make_cudaPitchedPtr(out_d, (Nz/2+1)*sizeof(C_t), Nz/2+1, dim.size_y[p_j]);
                        cpy_params.extent = make_cudaExtent((Nz/2+1)*sizeof(C_t), dim.size_y[p_j], dim.size_x[p_i]);
                        cpy_params.kind   = cudaMemcpyDeviceToHost;
                        
                        CUDA_CALL(cudaMemcpy3DAsync(&cpy_params));

                        send_count += dim.size_x[p_i]*dim.size_y[p_j]*(Nz/2+1);
                    } else {                    
                        cudaMemcpy3DParms cpy_params = {0};
                        cpy_params.srcPos = make_cudaPos(0, dim.start_y[p_j], dim.start_x[p_i]);
                        cpy_params.srcPtr = make_cudaPitchedPtr(complex, (Nz/2+1)*sizeof(C_t), Nz/2+1, Ny);
                        cpy_params.dstPos = make_cudaPos(0, 0, 0);
                        cpy_params.dstPtr = make_cudaPitchedPtr(&send_ptr[send_count], (Nz/2+1)*sizeof(C_t), Nz/2+1, dim.size_y[p_j]);
                        cpy_params.extent = make_cudaExtent((Nz/2+1)*sizeof(C_t), dim.size_y[p_j], dim.size_x[p_i]);
                        cpy_params.kind   = cudaMemcpyDeviceToHost;
                        
                        CUDA_CALL(cudaMemcpy3DAsync(&cpy_params));
                        CUDA_CALL(cudaDeviceSynchronize());
        
                        MPI_Isend(&send_ptr[send_count], dim.size_x[p_i]*dim.size_y[p_j]*(Nz/2+1)*sizeof(C_t), 
                            MPI_BYTE, p_i*P2+p_j, 0, MPI_COMM_WORLD, &send_req[p_i*P2+p_j]);
                        
                        send_count += dim.size_x[p_i]*dim.size_y[p_j]*(Nz/2+1);
                    }
    
                }
            }

            /************************************************************************************
            *
            *      Here, one could continue the previous computation with out_d for rank 0
            *
            *************************************************************************************/            

            MPI_Waitall(world_size, send_req.data(), MPI_STATUSES_IGNORE);
            timer->stop_store("Finished Send");
            timer->stop_store("Run complete");
            timer->gather();

        }
    } else {
        R_t *send_ptr;
        C_t *recv_ptr;

        MPI_Request send_req;
        MPI_Request recv_req;

        if (!cuda_aware) {
            CUDA_CALL(cudaMallocHost((void **)&send_ptr, dim.size_x[pidx_i]*dim.size_y[pidx_j]*Nz*sizeof(R_t)));
            CUDA_CALL(cudaMallocHost((void **)&recv_ptr, dim.size_x[pidx_i]*dim.size_y[pidx_j]*(Nz/2+1)*sizeof(C_t)));
        } else {
            send_ptr = in_d;
            recv_ptr = out_d;
        }
        CUDA_CALL(cudaDeviceSynchronize());
        MPI_Barrier(MPI_COMM_WORLD);
        timer->stop_store("init");

        for (int i = 0; i < runs; i++){
            // create new input data for each round (not included in benchmarked time)
            this->initializeRandArray(in_d, dim.size_x[pidx_i], dim.size_y[pidx_j]);
            // sync before each run
            MPI_Barrier(MPI_COMM_WORLD);
            timer->start();
    
            // send input data to rank 0 for global 3D FFT computation
            if (!cuda_aware) 
                CUDA_CALL(cudaMemcpy(send_ptr, in_d, dim.size_x[pidx_i]*dim.size_y[pidx_j]*Nz*sizeof(R_t), cudaMemcpyDeviceToHost));
    
            MPI_Isend(send_ptr, dim.size_x[pidx_i]*dim.size_y[pidx_j]*Nz*sizeof(R_t),
                MPI_BYTE, 0, 0, MPI_COMM_WORLD, &send_req);
    
            MPI_Wait(&send_req, MPI_STATUS_IGNORE);
            timer->stop_store("Finished Send");
    
            // wait for the computed result to arrive
            MPI_Irecv(recv_ptr, dim.size_x[pidx_i]*dim.size_y[pidx_j]*(Nz/2+1)*sizeof(C_t),
                MPI_BYTE, 0, 0, MPI_COMM_WORLD, &recv_req);
    
            MPI_Wait(&recv_req, MPI_STATUS_IGNORE);
            timer->stop_store("Finished Receive");
    
            if (!cuda_aware)
                CUDA_CALL(cudaMemcpy(out_d, recv_ptr, dim.size_x[pidx_i]*dim.size_y[pidx_j]*(Nz/2+1)*sizeof(C_t), cudaMemcpyHostToDevice));
            
            /*************************************************************************
            *
            *      Here, one could continue the previous computation with out_d
            *
            **************************************************************************/
            
            timer->stop_store("Run complete");
            timer->gather();
        }
    }
    //finalize
    MPI_Finalize();
    return 0;
}

template<typename T>
int Tests_Reference<T>::testcase1(const int opt, const int runs) {
    using R_t = typename cuFFT<T>::R_t;

    MPI_Init(NULL, NULL);

    //number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    //get global rank
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    R_t *in_d, *send_ptr, *recv_ptr, *out_d;

    CUDA_CALL(cudaMalloc((void **)&in_d, Nx*Ny*Nz*sizeof(R_t)));
    CUDA_CALL(cudaMalloc((void **)&out_d, (world_size-1)*Nx*Ny*Nz*sizeof(R_t)));

    if (!cuda_aware) {
        CUDA_CALL(cudaMallocHost((void **)&send_ptr, Nx*Ny*Nz*sizeof(R_t)));
        CUDA_CALL(cudaMallocHost((void **)&recv_ptr, (world_size-1)*Nx*Ny*Nz*sizeof(R_t)));
    } else {
        send_ptr = in_d;
        recv_ptr = out_d;
    }
    this->initializeRandArray(in_d, Nx, Ny);
    CUDA_CALL(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);

    std::vector<MPI_Request> send_req(world_size, MPI_REQUEST_NULL);
    std::vector<MPI_Request> recv_req(world_size, MPI_REQUEST_NULL);

    double t1, t2;
    if (opt == 0) {
        for (int i = 0; i < runs+10; i++) {   
            if (i == 10)
                t1 = MPI_Wtime();
            if (!cuda_aware) {
                CUDA_CALL(cudaMemcpyAsync(send_ptr, in_d, Nx*Ny*Nz*sizeof(R_t), cudaMemcpyDeviceToHost));
                CUDA_CALL(cudaDeviceSynchronize());
            }
            for (int p = 1; p < world_size; p++) {
                MPI_Isend(send_ptr, Nx*Ny*Nz*sizeof(R_t), MPI_BYTE, (rank+p)%world_size, (rank+p)%world_size, MPI_COMM_WORLD, &send_req[p]);
                MPI_Irecv(&recv_ptr[(p-1)*Nx*Ny*Nz], Nx*Ny*Nz*sizeof(R_t), MPI_BYTE, (rank+p)%world_size, rank, MPI_COMM_WORLD, &recv_req[p]);
            }
            MPI_Waitall(world_size, send_req.data(), MPI_STATUS_IGNORE);
            MPI_Waitall(world_size, recv_req.data(), MPI_STATUS_IGNORE);
            if (!cuda_aware){
                CUDA_CALL(cudaMemcpyAsync(out_d, recv_ptr, (world_size-1)*Nx*Ny*Nz*sizeof(R_t), cudaMemcpyHostToDevice));
                CUDA_CALL(cudaDeviceSynchronize());
            }
        }
        t2 = MPI_Wtime();
    } else if (opt == 1) {
        std::vector<int> sendcounts(world_size, Nx*Ny*Nz*sizeof(R_t));
        sendcounts[rank] = 0;
        std::vector<int> sdispls(world_size, 0);
        std::vector<int> recvcounts(world_size, Nx*Ny*Nz*sizeof(R_t));
        recvcounts[rank] = 0;
        std::vector<int> rdispls(world_size, 0);
        for (int i = 1; i < world_size; i++) {
            rdispls[(rank + i) % world_size] = (i-1)*Nx*Ny*Nz*sizeof(R_t);
        }

        for (int i = 0; i < runs+10; i++) {   
            if (i == 10)
                t1 = MPI_Wtime();
            if (!cuda_aware) {
                CUDA_CALL(cudaMemcpyAsync(send_ptr, in_d, Nx*Ny*Nz*sizeof(R_t), cudaMemcpyDeviceToHost));
                CUDA_CALL(cudaDeviceSynchronize());
            }

            MPI_Alltoallv(send_ptr, sendcounts.data(), sdispls.data(), MPI_BYTE, 
                recv_ptr, recvcounts.data(), rdispls.data(), MPI_BYTE, MPI_COMM_WORLD);

            if (!cuda_aware) {
                CUDA_CALL(cudaMemcpyAsync(out_d, recv_ptr, (world_size-1)*Nx*Ny*Nz*sizeof(R_t), cudaMemcpyHostToDevice));
                CUDA_CALL(cudaDeviceSynchronize());
            }
        }
        t2 = MPI_Wtime();
    }
    double size = (world_size-1)*Nx*Ny*Nz*sizeof(R_t)*1.0e-6;
    // bandwidth in MB/s
    double bandwidth = size*runs/(t2-t1);
    printf("Bandwidth for rank=%d, runs=%d, size=%f MB in MB/s: %f\n", rank, runs, size, bandwidth);

    MPI_Finalize();
    return 0;
}

struct Callback_Params_Base {
    std::mutex mutex;
    std::condition_variable cv;
    std::vector<int> comm_ready;
};

struct Callback_Params {
    Callback_Params_Base *base_params;

    int p;
};

struct Thread_Params {
    Callback_Params_Base *base_params;

    void* send_ptr;
    int world_size;
    int rank;
    size_t Nx, Ny, Nz;
    std::vector<int> &sizes_x;
    std::vector<int> &sizes_y;
    std::vector<int> &start_y;
};

static void MPIsend_Callback(void *data) {
  struct Callback_Params *params = (Callback_Params *)data;
  struct Callback_Params_Base *base_params = params->base_params;
  {
    std::lock_guard<std::mutex> lk(base_params->mutex);
    base_params->comm_ready.push_back(params->p);
  }
  base_params->cv.notify_one();
}

template <typename T>
static void MPIsend_Thread(Thread_Params &params, std::vector<MPI_Request> &send_req) {
  using R_t = typename cuFFT<T>::R_t;
  struct Callback_Params_Base *base_params = params.base_params;

  R_t *send_ptr = (R_t *) params.send_ptr;

  for (int i = 0; i <params.world_size-1; i++){
    std::unique_lock<std::mutex> lk(base_params->mutex);
    base_params->cv.wait(lk, [base_params]{return !base_params->comm_ready.empty();});

    int p = base_params->comm_ready.back();
    base_params->comm_ready.pop_back();
    size_t oslice = params.Nz*params.start_y[p]*params.sizes_x[params.rank];

    MPI_Isend(&send_ptr[params.Nz*params.start_y[p]*params.sizes_x[params.rank]], 
        params.Nz*params.sizes_y[p]*params.sizes_x[params.rank]*sizeof(R_t), 
        MPI_BYTE, p, p, MPI_COMM_WORLD, &send_req[p]);

    lk.unlock();
  }
}

template<typename T>
int Tests_Reference<T>::testcase2(const int opt, const int runs) {
    using R_t = typename cuFFT<T>::R_t;

    MPI_Init(NULL, NULL);

    //number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    //get global rank
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    R_t *in_d, *send_ptr, *recv_ptr, *out_d;

    std::vector<int> sizes_x(world_size, Nx/world_size);
    std::vector<int> start_x(world_size, 0);
    for (int i = 0; i < world_size; i++) {
        if (i < Nx % world_size)
            sizes_x[i]++;
        if (i > 0)
            start_x[i] = start_x[i-1] + sizes_x[i-1];
    }

    std::vector<int> sizes_y(world_size, Ny/world_size);
    std::vector<int> start_y(world_size, 0);
    for (int i = 0; i < world_size; i++) {
        if (i < Ny % world_size)
            sizes_y[i]++;
        if (i > 0)
            start_y[i] = start_y[i-1] + sizes_y[i-1];
    }

    CUDA_CALL(cudaMalloc((void **)&in_d, sizes_x[rank]*Ny*Nz*sizeof(R_t)));
    CUDA_CALL(cudaMalloc((void **)&out_d, Nz*sizes_y[rank]*Nx*sizeof(R_t)));

    if (!cuda_aware) {
        CUDA_CALL(cudaMallocHost((void **)&send_ptr, sizes_x[rank]*Ny*Nz*sizeof(R_t)));
        CUDA_CALL(cudaMallocHost((void **)&recv_ptr, Nz*sizes_y[rank]*Nx*sizeof(R_t)));
    } else {
        CUDA_CALL(cudaMalloc((void **)&send_ptr, sizes_x[rank]*Ny*Nz*sizeof(R_t)));
        recv_ptr = out_d;
    }
    this->initializeRandArray(in_d, sizes_x[rank], Ny);
    CUDA_CALL(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);

    std::vector<MPI_Request> send_req(world_size, MPI_REQUEST_NULL);
    std::vector<MPI_Request> recv_req(world_size, MPI_REQUEST_NULL);

    double t1, t2;
    if (opt == 0) {
        for (int i = 0; i < runs+10; i++) {   
            if (i == 10)
                t1 = MPI_Wtime();
    
            for (int j = 1; j < world_size; j++) {
                int p = (rank+j)%world_size;
                
                MPI_Irecv(&recv_ptr[Nz*sizes_y[rank]*start_x[p]], Nz*sizes_y[rank]*sizes_x[p]*sizeof(R_t), MPI_BYTE, p, rank, MPI_COMM_WORLD, &recv_req[p]);
    
                CUDA_CALL(cudaMemcpy2DAsync(&send_ptr[Nz*start_y[p]*sizes_x[rank]], Nz*sizes_y[p]*sizeof(R_t), 
                    &in_d[Nz*start_y[p]], Nz*Ny*sizeof(R_t), Nz*sizes_y[p]*sizeof(R_t), sizes_x[rank], cuda_aware?cudaMemcpyDeviceToDevice:cudaMemcpyDeviceToHost));
                CUDA_CALL(cudaDeviceSynchronize());
    
                MPI_Isend(&send_ptr[Nz*start_y[p]*sizes_x[rank]], Nz*sizes_y[p]*sizes_x[rank]*sizeof(R_t), MPI_BYTE, p, p, MPI_COMM_WORLD, &send_req[p]);
            }
    
            MPI_Waitall(world_size, send_req.data(), MPI_STATUS_IGNORE);
            MPI_Waitall(world_size, recv_req.data(), MPI_STATUS_IGNORE);
            if (!cuda_aware) {
                CUDA_CALL(cudaMemcpyAsync(out_d, recv_ptr, Nz*sizes_y[rank]*Nx*sizeof(R_t), cudaMemcpyHostToDevice));
                CUDA_CALL(cudaDeviceSynchronize());
            }
    
        }
        t2 = MPI_Wtime();
    } else if (opt == 1) {

        std::vector<cudaStream_t> streams(world_size);
        CUDA_CALL(cudaStreamCreate(&streams[0]));

        Callback_Params_Base base_params;
        std::vector<Callback_Params> params_array;

        for (int i = 1; i < world_size; i++){
            CUDA_CALL(cudaStreamCreate(&streams[1]));
            int p = (rank+i)%world_size;
            Callback_Params params = {&base_params, p};
            params_array.push_back(params);
        }

        Thread_Params thread_params = {&base_params, send_ptr, world_size, rank, Nx, Ny, Nz, sizes_x, sizes_y, start_y};

        for (int i = 0; i < runs+10; i++) {   
            if (i == 10)
                t1 = MPI_Wtime();
    
            for (int j = 1; j < world_size; j++) {
                int p = (rank+j)%world_size;
                
                MPI_Irecv(&recv_ptr[Nz*sizes_y[rank]*start_x[p]], Nz*sizes_y[rank]*sizes_x[p]*sizeof(R_t), MPI_BYTE, p, rank, MPI_COMM_WORLD, &recv_req[p]);
    
                CUDA_CALL(cudaMemcpy2DAsync(&send_ptr[Nz*start_y[p]*sizes_x[rank]], Nz*sizes_y[p]*sizeof(R_t), 
                    &in_d[Nz*start_y[p]], Nz*Ny*sizeof(R_t), Nz*sizes_y[p]*sizeof(R_t), sizes_x[rank], cuda_aware?cudaMemcpyDeviceToDevice:cudaMemcpyDeviceToHost, streams[p]));
                CUDA_CALL(cudaDeviceSynchronize());
    
                CUDA_CALL(cudaLaunchHostFunc(streams[p], MPIsend_Callback, (void *)&params_array[j-1]));
            }
            std::thread mpisend_thread(&MPIsend_Thread<T>, std::ref(thread_params), std::ref(send_req));
            MPI_Waitall(world_size, recv_req.data(), MPI_STATUS_IGNORE);
    
            mpisend_thread.join();
            MPI_Waitall(world_size, send_req.data(), MPI_STATUS_IGNORE);
            if (!cuda_aware) {
                CUDA_CALL(cudaMemcpyAsync(out_d, recv_ptr, Nz*sizes_y[rank]*Nx*sizeof(R_t), cudaMemcpyHostToDevice));
                CUDA_CALL(cudaDeviceSynchronize());
            }
    
        }
        t2 = MPI_Wtime();
    } else if (opt == 2) {
        std::vector<MPI_Datatype> MPI_PENCILS(world_size);
        for (int i = 0; i < world_size; i++) {
            MPI_Type_vector(sizes_x[rank], Nz*sizes_y[i]*sizeof(R_t), Nz*Ny*sizeof(R_t), MPI_BYTE, &MPI_PENCILS[i]);
            MPI_Type_commit(&MPI_PENCILS[i]);
        }

        for (int i = 0; i < runs+10; i++) {   
            if (i == 10)
                t1 = MPI_Wtime();
    
            if (!cuda_aware) {
                CUDA_CALL(cudaMemcpyAsync(send_ptr, in_d, Nz*Ny*sizes_x[rank]*sizeof(R_t), cudaMemcpyDeviceToHost));
                CUDA_CALL(cudaDeviceSynchronize());
            }

            for (int j = 1; j < world_size; j++) {
                int p = (rank+j)%world_size;
                MPI_Irecv(&recv_ptr[Nz*sizes_y[rank]*start_x[p]], Nz*sizes_y[rank]*sizes_x[p]*sizeof(R_t), MPI_BYTE, p, rank, MPI_COMM_WORLD, &recv_req[p]);
                MPI_Isend(&send_ptr[Nz*start_y[p]], 1, MPI_PENCILS[p], p, p, MPI_COMM_WORLD, &send_req[p]);
            }
            MPI_Waitall(world_size, send_req.data(), MPI_STATUS_IGNORE);
            MPI_Waitall(world_size, recv_req.data(), MPI_STATUS_IGNORE);

            if (!cuda_aware) {
                CUDA_CALL(cudaMemcpyAsync(out_d, recv_ptr, Nz*sizes_y[rank]*Nx*sizeof(R_t), cudaMemcpyHostToDevice));
                CUDA_CALL(cudaDeviceSynchronize());
            }
    
        }
        t2 = MPI_Wtime();
    }

    double size_in = Nz*sizes_y[rank]*(Nx-sizes_x[rank])*sizeof(R_t)*1.0e-6;
    double size_out = Nz*(Ny-sizes_y[rank])*sizes_x[rank]*sizeof(R_t)*1.0e-6;
    // bandwidth in MB/s
    double bandwidth_in = size_in*runs/(t2-t1);
    double bandwidth_out = size_in*runs/(t2-t1);
    printf("Incoming Bandwidth for rank=%d, runs=%d, size=%f MB in MB/s: %f\n", rank, runs, size_in, bandwidth_in);
    printf("Outgoing Bandwidth for rank=%d, runs=%d, size=%f MB in MB/s: %f\n", rank, runs, size_out, bandwidth_out);

    MPI_Finalize();
    return 0;
}

template class Tests_Reference<float>;
template class Tests_Reference<double>;