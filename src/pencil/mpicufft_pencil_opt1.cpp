#include "mpicufft_pencil_opt1.hpp"
#include "cufft.hpp"
#include "cuda_profiler_api.h"
#include <cuda_runtime.h>

#define error(e) {                  \
    throw std::runtime_error(e);    \
}

#define CUDA_CALL(x) do { if((x)!=cudaSuccess) {        \
    printf("Error %d at %s:%d\n",x,__FILE__,__LINE__);  \
    exit(EXIT_FAILURE); }} while(0)

#define CUFFT_CALL(x) do { if((x)!=CUFFT_SUCCESS) {     \
    printf("Error %d at %s:%d\n",x,__FILE__,__LINE__);  \
    exit(EXIT_FAILURE); }} while(0)

#define DEBUG 1
#define debug(d, v) {                                                 \
  if (DEBUG == 1) {                                                   \
    printf("DEBUG (%d,%d): %s: %s in %s:%d\n",pidx_i,pidx_j,d,v,__FILE__,__LINE__); \
  }                                                                   \
}

#define debug_h(v) {                                                  \
  if (DEBUG == 1) {                                                   \
    printf("%s",v);                \
  }                                                                   \
}

#define debug_int(d, v) {                                             \
  if (DEBUG == 1) {                                                   \
    printf("DEBUG (%d,%d): %s: %d in %s:%d\n",pidx_i,pidx_j,d,v,__FILE__,__LINE__); \
  }                                                                   \
}

#define debug_p(d, v) {                                                  \
  if (DEBUG == 1) {                                                   \
    printf("DEBUG (%d,%d): %s: %p in %s:%d\n",pidx_i,pidx_j,d,v,__FILE__,__LINE__); \
  }                                                                   \
}

template<typename T>
MPIcuFFT_Pencil_Opt1<T>::~MPIcuFFT_Pencil_Opt1() {
    CUFFT_CALL(cufftDestroy(planC2C_0));
}

template<typename T>
void MPIcuFFT_Pencil_Opt1<T>::initFFT(GlobalSize *global_size_, Partition *partition_, bool allocate) {
    global_size = global_size_;
    partition = partition_;

    mkdir((config.benchmark_dir +  "/pencil").c_str(), 0777);
    std::stringstream ss;
    ss << config.benchmark_dir <<  "/pencil/test_1_" << config.comm_method << "_" << config.send_method << "_" << global_size->Nx;
    ss << "_" << cuda_aware << "_" << partition->P1 << "_" << partition->P2 << ".csv";
    std::string filename = ss.str();
    timer = new Timer(comm, 0, pcnt, pidx, section_descriptions, filename);

    timer->start();

    // Determine pidx_x and pidx_y using partition
    if (partition == nullptr || global_size == nullptr)
        error("GlobalSize or Partition not initialized!");
    
    if (partition->P1 * partition->P2 != pcnt)
        error("Invalid Input Partition!");
    // pidx = pidx_i * P2 + pidx_j
    pidx_i = pidx/partition->P2;
    pidx_j = pidx%partition->P2;

    // Determine all Partition_Dimensions
    // input_dim:
    input_dim.size_x.resize(partition->P1, global_size->Nx/partition->P1);
    for (int i = 0; i < global_size->Nx%partition->P1; i++)
        input_dim.size_x[i]++;
    input_dim.size_y.resize(partition->P2, global_size->Ny/partition->P2);
    for (int j = 0; j < global_size->Ny%partition->P2; j++)
        input_dim.size_y[j]++;
    input_dim.size_z.resize(1, global_size->Nz);
    input_dim.computeOffsets();
    // transposed_dim:
    transposed_dim.size_x = input_dim.size_x;
    transposed_dim.size_y.resize(1, global_size->Ny);
    transposed_dim.size_z.resize(partition->P2, (global_size->Nz/2+1)/partition->P2);
    for (int k = 0; k < (global_size->Nz/2+1)%partition->P2; k++)
        transposed_dim.size_z[k]++;
    transposed_dim.computeOffsets();
    // output_dim:
    output_dim.size_x.resize(1, global_size->Nx);
    output_dim.size_y.resize(partition->P1, global_size->Ny/partition->P1);
    for (int j = 0; j < global_size->Ny%partition->P1; j++)
        output_dim.size_y[j]++;
    output_dim.size_z = transposed_dim.size_z;
    output_dim.computeOffsets();

    // For the first transpose, we need P2 streams; For the second transpose, we need P1 streams
    for (int i = 0; i < std::max(partition->P1, partition->P2); i++){
        cudaStream_t stream;
        CUDA_CALL(cudaStreamCreate(&stream));
        streams.push_back(stream);
    }

    // Create MPI_comms for first and second transpose
    MPI_Comm_split(comm, pidx_i, pidx_j, &comm1);
    MPI_Comm_split(comm, pidx_j, pidx_i, &comm2);

    // Set comm_order for first and second transpose
    comm_order1.clear();
    for (int j = 1; j < partition->P2; j++)
        comm_order1.push_back((pidx_j+j)%partition->P2);

    comm_order2.clear();
    for (int i = 1; i < partition->P1; i++)
        comm_order2.push_back((pidx_i+i)%partition->P1);

    // send and recv requests
    send_req.resize(std::max(partition->P1, partition->P2), MPI_REQUEST_NULL);
    recv_req.resize(std::max(partition->P1, partition->P2), MPI_REQUEST_NULL);

    // Make FFT plan
    using R_t = typename cuFFT<T>::R_t;
    using C_t = typename cuFFT<T>::C_t;

    size_t ws_r2c;

    CUFFT_CALL(cufftCreate(&planR2C));
    CUFFT_CALL(cufftSetAutoAllocation(planR2C, 0));

    // If fft3d is set, then only one mpi node is used. Thus the full FFT is computed here.
    if (fft3d) {
        CUFFT_CALL(cufftMakePlan3d(planR2C, global_size->Nx, global_size->Ny, global_size->Nz, cuFFT<T>::R2Ctype, &ws_r2c));
        fft_worksize = ws_r2c;
    } else { // Otherwise, we use pencil decomposition
        size_t ws_c2c_0, ws_c2c_1;
        long long int batch[3] = {static_cast<long long int>(input_dim.size_y[pidx_j]*input_dim.size_x[pidx_i]), 
            static_cast<long long int>(transposed_dim.size_z[pidx_j]*transposed_dim.size_x[pidx_i]), 
            static_cast<long long int>(output_dim.size_z[pidx_j]*output_dim.size_y[pidx_i])};

        CUFFT_CALL(cufftCreate(&planC2C_0));      
        CUFFT_CALL(cufftSetAutoAllocation(planC2C_0, 0));
        CUFFT_CALL(cufftCreate(&planC2C_1));      
        CUFFT_CALL(cufftSetAutoAllocation(planC2C_1, 0));

        long long int n[3] = {static_cast<long long int>(output_dim.size_x[0]), static_cast<long long int>(transposed_dim.size_y[0]), 
            static_cast<long long int>(input_dim.size_z[0])};

        long long int inembed[3] = {1, 1, 1};
        long long int onembed[3] = {static_cast<long long int>(output_dim.size_z[pidx_j]*output_dim.size_y[pidx_i]), 
            static_cast<long long int>(transposed_dim.size_z[pidx_j]*transposed_dim.size_x[pidx_i]),
            static_cast<long long int>(input_dim.size_y[pidx_j]*input_dim.size_x[pidx_i])};
        long long int idist[3] = {static_cast<long long int>(output_dim.size_x[0]), 
            static_cast<long long int>(transposed_dim.size_y[0]), 
            static_cast<long long int>(input_dim.size_z[0])};
        long long int odist[3] = {1, 1, 1};

        // (1) For the first 1D FFT (z-direction), we transform the coordinate system such that
        // the data is afterwards continuous in y-direction
        CUFFT_CALL(cufftMakePlanMany64(planR2C, 1, &n[2], //plan, rank, *n
            &inembed[2], inembed[2], idist[2], //*inembed, istride, idist
            &onembed[2], onembed[2], odist[2], //*onembed, ostride, odist
            cuFFT<T>::R2Ctype, batch[0], &ws_r2c)); //type, batch, worksize

        // (2) For the second 1D FFT (y-direction), we transform the coordinate system such that
        // the data is afterwards continuous in x-direction
        CUFFT_CALL(cufftMakePlanMany64(planC2C_0, 1, &n[1], //plan, rank, *n
            &inembed[1], inembed[1], idist[1], //*inembed, istride, idist
            &onembed[1], onembed[1], odist[1], //*onembed, ostride, odist
            cuFFT<T>::C2Ctype, batch[1], &ws_c2c_0)); //type, batch, worksize

        // (3) For the last 1D FFT (x-direction), we transform the coordinate system such that
        // the data is afterwards continuous in z-direction (same data alignment as the input data)
        CUFFT_CALL(cufftMakePlanMany64(planC2C_1, 1, &n[0], //plan, rank, *n
            &inembed[0], inembed[0], idist[0], //*inembed, istride, idist
            &onembed[0], onembed[0], odist[0], //*onembed, ostride, odist
            cuFFT<T>::C2Ctype, batch[2], &ws_c2c_1)); //type, batch, worksize

        fft_worksize = std::max(ws_r2c, ws_c2c_0);
        fft_worksize = std::max(fft_worksize, ws_c2c_1);   
    }

    // Max space for the process's pencil. Needed for send/recv buffer
    domainsize = std::max(input_dim.size_x[pidx_i]*input_dim.size_y[pidx_j]*(input_dim.size_z[0]/2+1), 
        transposed_dim.size_x[pidx_i]*transposed_dim.size_y[0]*transposed_dim.size_z[pidx_j]);
    domainsize = sizeof(C_t) * std::max(domainsize, output_dim.size_x[0]*output_dim.size_y[pidx_i]*output_dim.size_z[pidx_j]);

    if (fft_worksize < domainsize)
        fft_worksize = domainsize;

    // worksize_d is split into 3 parts (2 if not cuda aware):
    // 1. temp buffer, for local transpose (if cuda aware) or buffer for host->device transfer of recv buffer
    // 2. recv buffer (if cuda aware)
    // 3. actual workspace (second slot if not cuda aware)
    // Note that we do not have to use a send buffer, if MPI is cuda_aware ~> reduced memory space when compared to the default variant
    worksize_d = fft_worksize + (fft3d ? 0 : (cuda_aware ? 2*domainsize : domainsize ));
    // if not cuda aware, then recv and send buffer are on the host side
    worksize_h = (cuda_aware || fft3d ? 0 : 2*domainsize);

    if (allocate)
        this->setWorkArea();

    if (config.comm_method == Peer2Peer) {
        if (config.send_method == Streams && !cuda_aware) {
            for (int i = 0; i < comm_order1.size(); i++){
                size_t p_j = comm_order1[i];
                Callback_Params params = {&base_params, p_j};
                params_array1.push_back(params);
            }
        } else if (config.send_method == MPI_Type) {
            MPI_RECV1 = std::vector<MPI_Datatype>(partition->P2);
            for (int i = 0; i < partition->P2; i++) {
                MPI_Type_vector(transposed_dim.size_z[pidx_j]*transposed_dim.size_x[pidx_i], input_dim.size_y[i]*sizeof(C_t),
                    global_size->Ny*sizeof(C_t), MPI_BYTE, &MPI_RECV1[i]);
                MPI_Type_commit(&MPI_RECV1[i]);
            }
        }
    } else {
        if (config.send_method == MPI_Type) {
            MPI_RECV1 = std::vector<MPI_Datatype>(partition->P2);
            MPI_SND1 = std::vector<MPI_Datatype>(partition->P2);
            sendcounts1 = std::vector<int>(partition->P2, 0);
            sdispls1 = std::vector<int>(partition->P2, 0);
            recvcounts1 = std::vector<int>(partition->P2, 1);
            rdispls1 = std::vector<int>(partition->P2, 0);
            for (int p = 0; p < partition->P2; p++) {
                sendcounts1[p] = sizeof(C_t)*transposed_dim.size_z[p]*input_dim.size_y[pidx_j]*input_dim.size_x[pidx_i];
                sdispls1[p] = sizeof(C_t)*transposed_dim.start_z[p]*input_dim.size_y[pidx_j]*input_dim.size_x[pidx_i];
                rdispls1[p] = sizeof(C_t)*input_dim.start_y[p];
                MPI_Type_vector(transposed_dim.size_z[pidx_j]*transposed_dim.size_x[pidx_i], input_dim.size_y[p]*sizeof(C_t),
                    global_size->Ny*sizeof(C_t), MPI_BYTE, &MPI_RECV1[p]);
                MPI_Type_commit(&MPI_RECV1[p]);
                MPI_SND1[p] = MPI_BYTE;                
            }
        } else {
            sendcounts1 = std::vector<int>(partition->P2, 0);
            sdispls1 = std::vector<int>(partition->P2, 0);
            recvcounts1 = std::vector<int>(partition->P2, 0);
            rdispls1 = std::vector<int>(partition->P2, 0);
            for (int p = 0; p < partition->P2; p++) {
                sendcounts1[p] = sizeof(C_t)*transposed_dim.size_z[p]*input_dim.size_y[pidx_j]*input_dim.size_x[pidx_i];
                sdispls1[p] = sizeof(C_t)*transposed_dim.start_z[p]*input_dim.size_y[pidx_j]*input_dim.size_x[pidx_i];
                recvcounts1[p] = sizeof(C_t)*transposed_dim.size_x[pidx_i]*input_dim.size_y[p]*transposed_dim.size_z[pidx_j];
                rdispls1[p] = sizeof(C_t)*transposed_dim.size_x[pidx_i]*input_dim.start_y[p]*transposed_dim.size_z[pidx_j];
            }
        }
    }

    if (config.comm_method2 == Peer2Peer) {
        if (config.send_method2 == Streams && !cuda_aware) {
            for (int i = 0; i < partition->P1; i++){
                size_t p_i = comm_order2[i];
                Callback_Params params = {&base_params, p_i};
                params_array2.push_back(params);
            }
        } else if (config.send_method2 == MPI_Type) {
            MPI_RECV2 = std::vector<MPI_Datatype>(partition->P1);
            for (int i = 0; i < partition->P1; i++) {
                MPI_Type_vector(output_dim.size_z[pidx_j]*output_dim.size_y[pidx_i], transposed_dim.size_x[i]*sizeof(C_t),
                    output_dim.size_x[0]*sizeof(C_t), MPI_BYTE, &MPI_RECV2[i]);
                MPI_Type_commit(&MPI_RECV2[i]);
            }
        }
    } else {
        if (config.send_method2 == MPI_Type) {
            MPI_RECV2 = std::vector<MPI_Datatype>(partition->P1);
            MPI_SND2 = std::vector<MPI_Datatype>(partition->P1);
            sendcounts2 = std::vector<int>(partition->P1, 0);
            sdispls2 = std::vector<int>(partition->P1, 0);
            recvcounts2 = std::vector<int>(partition->P1, 1);
            rdispls2 = std::vector<int>(partition->P1, 0);
            for (int p = 0; p < partition->P1; p++) {
                sendcounts2[p] = sizeof(C_t)*transposed_dim.size_x[pidx_i]*transposed_dim.size_z[pidx_j]*output_dim.size_y[p];
                sdispls2[p] = sizeof(C_t)*transposed_dim.size_x[pidx_i]*transposed_dim.size_z[pidx_j]*output_dim.start_y[p];
                rdispls2[p] = sizeof(C_t)*transposed_dim.start_x[p];
                MPI_Type_vector(output_dim.size_z[pidx_j]*output_dim.size_y[pidx_i], transposed_dim.size_x[p]*sizeof(C_t),
                    output_dim.size_x[0]*sizeof(C_t), MPI_BYTE, &MPI_RECV2[p]);
                MPI_Type_commit(&MPI_RECV2[p]);
                MPI_SND2[p] = MPI_BYTE;
            }
        } else {
            sendcounts2 = std::vector<int>(partition->P1, 0);
            sdispls2 = std::vector<int>(partition->P1, 0);
            recvcounts2 = std::vector<int>(partition->P1, 0);
            rdispls2 = std::vector<int>(partition->P1, 0);
            for (int p = 0; p < partition->P1; p++) {
                sendcounts2[p] = sizeof(C_t)*transposed_dim.size_x[pidx_i]*transposed_dim.size_z[pidx_j]*output_dim.size_y[p];
                sdispls2[p] = sizeof(C_t)*transposed_dim.size_x[pidx_i]*transposed_dim.size_z[pidx_j]*output_dim.start_y[p];
                recvcounts2[p] = sizeof(C_t)*transposed_dim.size_x[p]*output_dim.size_y[pidx_i]*output_dim.size_z[pidx_j];
                rdispls2[p] = sizeof(C_t)*transposed_dim.start_x[p]*output_dim.size_y[pidx_i]*output_dim.size_z[pidx_j];
            }
        }
    }

    CUDA_CALL(cudaDeviceSynchronize());
    timer->stop_store("init");
}

template<typename T>
void MPIcuFFT_Pencil_Opt1<T>::setWorkArea(void *device, void *host){
    if (!domainsize)
        error("cuFFT plans are not yet initialized!");

    if (device && allocated_d) {
        CUDA_CALL(cudaFree(workarea_d));
        allocated_d = false;
        workarea_d = device;
    } else if (device && !allocated_d){
        workarea_d = device;
    } else if (!device && !allocated_d){
        CUDA_CALL(cudaMalloc(&workarea_d, worksize_d));
        allocated_d = true;
    }

    // mem_d stores pointer to all (above described) workspaces (temp, recv, actual)
    mem_d.clear();
    for (size_t i = 0; i < 1 + (fft3d ? 0 : (cuda_aware ? 2 : 1)); i++)
        mem_d.push_back(&static_cast<char*>(workarea_d)[i*domainsize]);
    
    if (fft3d) {
        CUFFT_CALL(cufftSetWorkArea(planR2C, mem_d[0]));
    } else if (!fft3d) {
        size_t offset = cuda_aware ? 2 : 1;

        // Since each computation R2C, C2C_0 and C2C_1 is split by a global transpose,
        // the same workspace can be reused
        CUFFT_CALL(cufftSetWorkArea(planR2C, mem_d[offset]));
        // Note that we had to split the 1D FFT in y-direction into multiple streams for the default option
        // Due to the coordinate transformation, we are able to use a single cuFFT-plan
        CUFFT_CALL(cufftSetWorkArea(planC2C_0, mem_d[offset]));
        CUFFT_CALL(cufftSetWorkArea(planC2C_1, mem_d[offset]));
    }

    // Same proceeding for the host. But only if cuda_aware == false.
    // In this case, host memory is used for send and recv buffer
    if (worksize_h){
        if (host && allocated_h){
            CUDA_CALL(cudaFreeHost(workarea_h));
            allocated_h = false;
            workarea_h = host;
        } else if (host && !allocated_h){
            workarea_h = host;
        } else if (!host && !allocated_h) {
            CUDA_CALL(cudaMallocHost(&workarea_h, worksize_h));
            allocated_h = true;
        }

        mem_h.clear();
        for (size_t i = 0; i < 2; i++)
            mem_h.push_back(&static_cast<char*>(workarea_h)[i*domainsize]);
    }

    initialized = true;
}


/* ***********************************************************************************************************************
*                                          Helper Methods for First Global Transpose
*  *********************************************************************************************************************** *
*                                                           Peer2Peer
*  *********************************************************************************************************************** */

template<typename T>
void MPIcuFFT_Pencil_Opt1<T>::Peer2Peer_Sync_FirstTranspose(void *complex_, void *recv_ptr_) {
    printf("1. Peer2Peer Sync\n");
    using C_t = typename cuFFT<T>::C_t;
    C_t *complex = cuFFT<T>::complex(complex_);
    C_t *recv_ptr = cuFFT<T>::complex(recv_ptr_);
    C_t *send_ptr;

    if (cuda_aware) {
        send_ptr = complex;
    } else {
        send_ptr = cuFFT<T>::complex(mem_h[1]);
        CUDA_CALL(cudaMemcpyAsync(send_ptr, complex, 
            sizeof(C_t)*(input_dim.size_z[0]/2+1)*input_dim.size_y[pidx_j]*input_dim.size_x[pidx_i], cudaMemcpyDeviceToHost));
        CUDA_CALL(cudaDeviceSynchronize());
    }

    for (size_t i = 0; i < comm_order1.size(); i++){
        size_t p_j = comm_order1[i];

        // Start non-blocking MPI recv
        MPI_Irecv(&recv_ptr[transposed_dim.size_x[pidx_i]*input_dim.start_y[p_j]*transposed_dim.size_z[pidx_j]],
            sizeof(C_t)*transposed_dim.size_x[pidx_i]*input_dim.size_y[p_j]*transposed_dim.size_z[pidx_j], MPI_BYTE,
            p_j, p_j, comm1, &recv_req[p_j]);

        size_t oslice = transposed_dim.start_z[p_j]*input_dim.size_y[pidx_j]*input_dim.size_x[pidx_i];

        if (i == 0)
            timer->stop_store("First Transpose (First Send)");

        MPI_Isend(&send_ptr[oslice], 
            sizeof(C_t)*transposed_dim.size_z[p_j]*input_dim.size_y[pidx_j]*input_dim.size_x[pidx_i], MPI_BYTE,
            p_j, pidx_j, comm1, &send_req[p_j]);       
    }
} 

template<typename T>
void MPIcuFFT_Pencil_Opt1<T>::Peer2Peer_Streams_FirstTranspose(void *complex_, void *recv_ptr_) {
    printf("1. Peer2Peer Streams\n");
    using C_t = typename cuFFT<T>::C_t;
    C_t *complex = cuFFT<T>::complex(complex_);
    C_t *recv_ptr = cuFFT<T>::complex(recv_ptr_);
    C_t *send_ptr;

    if (!cuda_aware) {
        send_ptr = cuFFT<T>::complex(mem_h[1]);
        mpisend_thread1 = std::thread(&MPIcuFFT_Pencil_Opt1<T>::MPIsend_Thread_FirstCallback, this, std::ref(base_params), send_ptr);
    }

    for (size_t i = 0; i < comm_order1.size(); i++){
        size_t p_j = comm_order1[i];

        // Start non-blocking MPI recv
        MPI_Irecv(&recv_ptr[transposed_dim.size_x[pidx_i]*input_dim.start_y[p_j]*transposed_dim.size_z[pidx_j]],
            sizeof(C_t)*transposed_dim.size_x[pidx_i]*input_dim.size_y[p_j]*transposed_dim.size_z[pidx_j], MPI_BYTE,
            p_j, p_j, comm1, &recv_req[p_j]);

        size_t oslice = transposed_dim.start_z[p_j]*input_dim.size_y[pidx_j]*input_dim.size_x[pidx_i];
        if (!cuda_aware) {
            // TODO: Add option, where we have single memcpy before the loop
            // ~> Pro: Only single memcpy; Con: Additional Sync needed
            CUDA_CALL(cudaMemcpyAsync(&send_ptr[oslice], &complex[oslice],
                sizeof(C_t)*transposed_dim.size_z[p_j]*input_dim.size_y[pidx_j]*input_dim.size_x[pidx_i], 
                cudaMemcpyDeviceToHost, streams[p_j]));

            // After copy is complete, MPI starts a non-blocking send operation
            CUDA_CALL(cudaLaunchHostFunc(streams[p_j], this->MPIsend_Callback, (void *)&params_array1[i]));
        } else {
            if (i == 0)
                timer->stop_store("First Transpose (First Send)");

            MPI_Isend(&complex[oslice], 
                sizeof(C_t)*transposed_dim.size_z[p_j]*input_dim.size_y[pidx_j]*input_dim.size_x[pidx_i], MPI_BYTE,
                p_j, pidx_j, comm1, &send_req[p_j]);
        }            
    }
} 

template<typename T>
void MPIcuFFT_Pencil_Opt1<T>::Peer2Peer_MPIType_FirstTranspose(void *complex_, void *recv_ptr_) {
    printf("1. Peer2Peer MPI_Type\n");
    using C_t = typename cuFFT<T>::C_t;
    C_t *complex = cuFFT<T>::complex(complex_);
    C_t *recv_ptr = cuFFT<T>::complex(recv_ptr_);
    C_t *send_ptr;

    if (cuda_aware) {
        send_ptr = complex;
    } else {
        send_ptr = cuFFT<T>::complex(mem_h[1]);
        CUDA_CALL(cudaMemcpyAsync(send_ptr, complex, 
            sizeof(C_t)*(input_dim.size_z[0]/2+1)*input_dim.size_y[pidx_j]*input_dim.size_x[pidx_i], cudaMemcpyDeviceToHost));
        CUDA_CALL(cudaDeviceSynchronize());
    }

    for (size_t i = 0; i < comm_order1.size(); i++){
        size_t p_j = comm_order1[i];

        // Start non-blocking MPI recv
        MPI_Irecv(&recv_ptr[input_dim.start_y[p_j]], 1, MPI_RECV1[p_j], p_j, p_j, comm1, &recv_req[p_j]);

        size_t oslice = transposed_dim.start_z[p_j]*input_dim.size_y[pidx_j]*input_dim.size_x[pidx_i];

        if (i == 0)
            timer->stop_store("First Transpose (First Send)");

        MPI_Isend(&send_ptr[oslice], 
            sizeof(C_t)*transposed_dim.size_z[p_j]*input_dim.size_y[pidx_j]*input_dim.size_x[pidx_i], MPI_BYTE,
            p_j, pidx_j, comm1, &send_req[p_j]);       
    }
} 

template<typename T>
void MPIcuFFT_Pencil_Opt1<T>::Peer2Peer_Communication_FirstTranspose(void *complex_) {
    using C_t = typename cuFFT<T>::C_t;
    C_t *complex = cuFFT<T>::complex(complex_);
    C_t *recv_ptr, *temp_ptr; 

    temp_ptr = cuFFT<T>::complex(mem_d[0]);

    if (config.send_method == MPI_Type) {
        if (cuda_aware)
            recv_ptr = temp_ptr;
        else 
            recv_ptr = cuFFT<T>::complex(mem_h[0]);

        this->Peer2Peer_MPIType_FirstTranspose(complex_, (void *)recv_ptr);

        cudaMemcpy3DParms cpy_params = {0};
        cpy_params.srcPos = make_cudaPos(0, 0, transposed_dim.start_z[pidx_j]);
        cpy_params.srcPtr = make_cudaPitchedPtr(complex, input_dim.size_y[pidx_j]*sizeof(C_t), input_dim.size_y[pidx_j], input_dim.size_x[pidx_i]);
        cpy_params.dstPos = make_cudaPos(input_dim.start_y[pidx_j]*sizeof(C_t), 0, 0);
        cpy_params.dstPtr = make_cudaPitchedPtr(recv_ptr, global_size->Ny*sizeof(C_t), global_size->Ny, transposed_dim.size_x[pidx_i]);
        cpy_params.extent = make_cudaExtent(input_dim.size_y[pidx_j]*sizeof(C_t), transposed_dim.size_x[pidx_i], transposed_dim.size_z[pidx_j]);
        cpy_params.kind   = cuda_aware?cudaMemcpyDeviceToDevice:cudaMemcpyDeviceToHost;

        CUDA_CALL(cudaMemcpy3DAsync(&cpy_params, streams[pidx_j]));

        timer->stop_store("Transpose (Start Receive)");
        MPI_Waitall(partition->P2, recv_req.data(), MPI_STATUSES_IGNORE);

        if (!cuda_aware) {
            CUDA_CALL(cudaMemcpyAsync(temp_ptr, recv_ptr, 
                sizeof(C_t)*transposed_dim.size_y[0]*transposed_dim.size_x[pidx_i]*transposed_dim.size_z[pidx_j], cudaMemcpyHostToDevice));
        }
        CUDA_CALL(cudaDeviceSynchronize());
    } else {
        if (cuda_aware)
            recv_ptr = cuFFT<T>::complex(mem_d[1]);
        else 
            recv_ptr = cuFFT<T>::complex(mem_h[0]);

        if (config.send_method == Sync) 
            this->Peer2Peer_Sync_FirstTranspose(complex_, (void *)recv_ptr);
        else 
            this->Peer2Peer_Streams_FirstTranspose(complex_, (void *)recv_ptr);

        // Transpose local block and copy it to the temp buffer
        timer->stop_store("First Transpose (Start Local Transpose)");
        {
            cudaMemcpy3DParms cpy_params = {0};
            cpy_params.srcPos = make_cudaPos(0, 0, transposed_dim.start_z[pidx_j]);
            cpy_params.srcPtr = make_cudaPitchedPtr(complex, input_dim.size_y[pidx_j]*sizeof(C_t), input_dim.size_y[pidx_j], input_dim.size_x[pidx_i]);
            cpy_params.dstPos = make_cudaPos(input_dim.start_y[pidx_j]*sizeof(C_t), 0, 0);
            cpy_params.dstPtr = make_cudaPitchedPtr(temp_ptr, global_size->Ny*sizeof(C_t), global_size->Ny, transposed_dim.size_x[pidx_i]);
            cpy_params.extent = make_cudaExtent(input_dim.size_y[pidx_j]*sizeof(C_t), transposed_dim.size_x[pidx_i], transposed_dim.size_z[pidx_j]);
            cpy_params.kind   = cudaMemcpyDeviceToDevice;

            CUDA_CALL(cudaMemcpy3DAsync(&cpy_params, streams[pidx_j]));
        }

        // Start copying the received blocks to the temp buffer, where the second 1D FFT (y-direction) can be computed
        // Since the received data has to be realigned (independent of cuda_aware), we use cudaMemcpy3D.
        timer->stop_store("First Transpose (Start Receive)");
        int p;
        do {
            // recv_req contains one null handle (i.e. recv_req[pidx_j]) and P2-1 active handles
            // If all active handles are processed, Waitany will return MPI_UNDEFINED
            MPI_Waitany(partition->P2, recv_req.data(), &p, MPI_STATUSES_IGNORE);
            if (p == MPI_UNDEFINED)
                break;

            // At this point, we received data of one of the P2-1 other relevant processes
            cudaMemcpy3DParms cpy_params = {0};
            cpy_params.srcPos = make_cudaPos(0, 0, 0);
            cpy_params.srcPtr = make_cudaPitchedPtr(&recv_ptr[transposed_dim.size_x[pidx_i]*input_dim.start_y[p]*transposed_dim.size_z[pidx_j]],
                input_dim.size_y[p]*sizeof(C_t), input_dim.size_y[p], input_dim.size_x[pidx_i]);
            cpy_params.dstPos = make_cudaPos(input_dim.start_y[p]*sizeof(C_t), 0, 0);
            cpy_params.dstPtr = make_cudaPitchedPtr(temp_ptr, global_size->Ny*sizeof(C_t), global_size->Ny, transposed_dim.size_x[pidx_i]);
            cpy_params.extent = make_cudaExtent(input_dim.size_y[p]*sizeof(C_t), input_dim.size_x[pidx_i], transposed_dim.size_z[pidx_j]);
            cpy_params.kind   = cuda_aware ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice;

            CUDA_CALL(cudaMemcpy3DAsync(&cpy_params, streams[(pidx_j + partition->P2 - p) % partition->P2]));   // TODO: check if this is the best stream for selection!
        } while (p != MPI_UNDEFINED);
        // For the 1D FFT in y-direction, all data packages have to be received
        CUDA_CALL(cudaDeviceSynchronize());
        timer->stop_store("First Transpose (Finished Receive)");
    }
} 


/* ***********************************************************************************************************************
*                                          Helper Methods for First Global Transpose
*  *********************************************************************************************************************** *
*                                                           All2All
*  *********************************************************************************************************************** */

template<typename T>
void MPIcuFFT_Pencil_Opt1<T>::All2All_Sync_FirstTranspose(void* complex_) {
    printf("1. All2All Sync\n");
    using C_t = typename cuFFT<T>::C_t;
    C_t *complex = cuFFT<T>::complex(complex_);
    C_t *send_ptr, *recv_ptr, *temp_ptr;
    temp_ptr = cuFFT<T>::complex(mem_d[0]);
    if (cuda_aware) {
        recv_ptr = cuFFT<T>::complex(mem_d[1]);
        send_ptr = complex;
    } else {
        recv_ptr = cuFFT<T>::complex(mem_h[0]);
        send_ptr = cuFFT<T>::complex(mem_h[1]);
        CUDA_CALL(cudaMemcpyAsync(send_ptr, complex, 
            sizeof(C_t)*(input_dim.size_z[0]/2+1)*input_dim.size_y[pidx_j]*input_dim.size_x[pidx_i], cudaMemcpyDeviceToHost));
        CUDA_CALL(cudaDeviceSynchronize());
    }

    timer->stop_store("Transpose (Start All2All)");
    MPI_Alltoallv(send_ptr, sendcounts1.data(), sdispls1.data(), MPI_BYTE, 
                recv_ptr, recvcounts1.data(), rdispls1.data(), MPI_BYTE, comm1);
    timer->stop_store("Transpose (Finished All2All)");

    for (int p = 0; p < partition->P2; p++) {
        // At this point, we received data of one of the P2-1 other relevant processes
        cudaMemcpy3DParms cpy_params = {0};
        cpy_params.srcPos = make_cudaPos(0, 0, 0);
        cpy_params.srcPtr = make_cudaPitchedPtr(&recv_ptr[transposed_dim.size_x[pidx_i]*input_dim.start_y[p]*transposed_dim.size_z[pidx_j]],
            input_dim.size_y[p]*sizeof(C_t), input_dim.size_y[p], input_dim.size_x[pidx_i]);
        cpy_params.dstPos = make_cudaPos(input_dim.start_y[p]*sizeof(C_t), 0, 0);
        cpy_params.dstPtr = make_cudaPitchedPtr(temp_ptr, global_size->Ny*sizeof(C_t), global_size->Ny, transposed_dim.size_x[pidx_i]);
        cpy_params.extent = make_cudaExtent(input_dim.size_y[p]*sizeof(C_t), input_dim.size_x[pidx_i], transposed_dim.size_z[pidx_j]);
        cpy_params.kind   = cuda_aware ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice;

        CUDA_CALL(cudaMemcpy3DAsync(&cpy_params, streams[p]));   // TODO: check if this is the best stream for selection!
    }
    CUDA_CALL(cudaDeviceSynchronize());
}

template<typename T>
void MPIcuFFT_Pencil_Opt1<T>::All2All_MPIType_FirstTranspose(void* complex_) {
    printf("1. All2All MPI_Type\n");
    using C_t = typename cuFFT<T>::C_t;
    C_t *complex = cuFFT<T>::complex(complex_);
    C_t *send_ptr, *recv_ptr, *temp_ptr;
    temp_ptr = cuFFT<T>::complex(mem_d[0]);
    if (cuda_aware) {
        recv_ptr = temp_ptr;
        send_ptr = complex;
    } else {
        recv_ptr = cuFFT<T>::complex(mem_h[0]);
        send_ptr = cuFFT<T>::complex(mem_h[1]);
        CUDA_CALL(cudaMemcpyAsync(send_ptr, complex, 
            sizeof(C_t)*(input_dim.size_z[0]/2+1)*input_dim.size_y[pidx_j]*input_dim.size_x[pidx_i], cudaMemcpyDeviceToHost));
        CUDA_CALL(cudaDeviceSynchronize());
    }

    timer->stop_store("Transpose (Start All2All)");
    MPI_Alltoallw(send_ptr, sendcounts1.data(), sdispls1.data(), MPI_SND1.data(), 
                recv_ptr, recvcounts1.data(), rdispls1.data(), MPI_RECV1.data(), comm1);
    timer->stop_store("Transpose (Finished All2All)");

    if (!cuda_aware) {
        CUDA_CALL(cudaMemcpyAsync(temp_ptr, recv_ptr, 
            transposed_dim.size_y[0]*transposed_dim.size_x[pidx_i]*transposed_dim.size_z[pidx_j]*sizeof(C_t), cudaMemcpyHostToDevice));
        CUDA_CALL(cudaDeviceSynchronize());
    }
}

template<typename T>
void MPIcuFFT_Pencil_Opt1<T>::All2All_Communication_FirstTranspose(void* complex_) {
    if (config.send_method == Sync)
        this->All2All_Sync_FirstTranspose(complex_);
    else   
        this->All2All_MPIType_FirstTranspose(complex_);
}

/* ***********************************************************************************************************************
*                                          Helper Methods for Second Global Transpose
*  *********************************************************************************************************************** *
*                                                           Peer2Peer
*  *********************************************************************************************************************** */

template<typename T>
void MPIcuFFT_Pencil_Opt1<T>::Peer2Peer_Sync_SecondTranspose(void *complex_, void* recv_ptr_) {
    printf("2. Peer2Peer Sync\n");
    using C_t = typename cuFFT<T>::C_t;
    C_t *complex = cuFFT<T>::complex(complex_);
    C_t *recv_ptr = cuFFT<T>::complex(recv_ptr_);
    C_t *send_ptr;

    if (cuda_aware) {
        send_ptr = complex;
    } else {
        send_ptr = cuFFT<T>::complex(mem_h[1]);
        CUDA_CALL(cudaMemcpyAsync(send_ptr, complex, 
            sizeof(C_t)*transposed_dim.size_y[0]*transposed_dim.size_x[pidx_i]*transposed_dim.size_z[pidx_j], cudaMemcpyDeviceToHost));
        CUDA_CALL(cudaDeviceSynchronize());
    }

    for (size_t i = 0; i < comm_order2.size(); i++){
        size_t p_i = comm_order2[i];

        // Start non-blocking MPI recv
        MPI_Irecv(&recv_ptr[transposed_dim.start_x[p_i]*output_dim.size_y[pidx_i]*output_dim.size_z[pidx_j]], 
            sizeof(C_t)*transposed_dim.size_x[p_i]*output_dim.size_y[pidx_i]*output_dim.size_z[pidx_j], MPI_BYTE,
            p_i, p_i, comm2, &recv_req[p_i]);

        size_t oslice = transposed_dim.size_x[pidx_i]*transposed_dim.size_z[pidx_j]*output_dim.start_y[p_i];

        if (i == 0)
            timer->stop_store("Second Transpose (First Send)");

        MPI_Isend(&send_ptr[oslice], sizeof(C_t)*transposed_dim.size_x[pidx_i]*transposed_dim.size_z[pidx_j]*output_dim.size_y[p_i],
            MPI_BYTE, p_i, pidx_i, comm2, &send_req[p_i]);
    }
}

template<typename T>
void MPIcuFFT_Pencil_Opt1<T>::Peer2Peer_Streams_SecondTranspose(void *complex_, void* recv_ptr_) {
    printf("1. Peer2Peer Streams\n");

    using C_t = typename cuFFT<T>::C_t;
    C_t *complex = cuFFT<T>::complex(complex_);
    C_t *recv_ptr = cuFFT<T>::complex(recv_ptr_);
    C_t *send_ptr;

    if (!cuda_aware) {
        send_ptr = cuFFT<T>::complex(mem_h[1]);
        mpisend_thread2 = std::thread(&MPIcuFFT_Pencil_Opt1<T>::MPIsend_Thread_SecondCallback, this, std::ref(base_params), send_ptr);
    }

    for (size_t i = 0; i < comm_order2.size(); i++){
        size_t p_i = comm_order2[i];

        // Start non-blocking MPI recv
        MPI_Irecv(&recv_ptr[transposed_dim.start_x[p_i]*output_dim.size_y[pidx_i]*output_dim.size_z[pidx_j]], 
            sizeof(C_t)*transposed_dim.size_x[p_i]*output_dim.size_y[pidx_i]*output_dim.size_z[pidx_j], MPI_BYTE,
            p_i, p_i, comm2, &recv_req[p_i]);

        size_t oslice = transposed_dim.size_x[pidx_i]*transposed_dim.size_z[pidx_j]*output_dim.start_y[p_i];
        if (!cuda_aware) {
            CUDA_CALL(cudaMemcpyAsync(&send_ptr[oslice], &complex[oslice], 
                sizeof(C_t)*transposed_dim.size_x[pidx_i]*transposed_dim.size_z[pidx_j]*output_dim.size_y[p_i], 
                cudaMemcpyDeviceToHost, streams[p_i]));

            // After copy is complete, MPI starts a non-blocking send operation
            CUDA_CALL(cudaLaunchHostFunc(streams[p_i], this->MPIsend_Callback, (void *)&params_array2[i]));
        } else {
            if (i == 0)
                timer->stop_store("Second Transpose (First Send)");

            MPI_Isend(&complex[oslice], sizeof(C_t)*transposed_dim.size_x[pidx_i]*transposed_dim.size_z[pidx_j]*output_dim.size_y[p_i],
                MPI_BYTE, p_i, pidx_i, comm2, &send_req[p_i]);
        }
    }   
}

template<typename T>
void MPIcuFFT_Pencil_Opt1<T>::Peer2Peer_MPIType_SecondTranspose(void *complex_, void* recv_ptr_) {
    printf("2. Peer2Peer MPI_Type\n");
    using C_t = typename cuFFT<T>::C_t;
    C_t *complex = cuFFT<T>::complex(complex_);
    C_t *recv_ptr = cuFFT<T>::complex(recv_ptr_);
    C_t *send_ptr;

    if (cuda_aware) {
        send_ptr = complex;
    } else {
        send_ptr = cuFFT<T>::complex(mem_h[1]);
        CUDA_CALL(cudaMemcpyAsync(send_ptr, complex, 
            sizeof(C_t)*transposed_dim.size_y[0]*transposed_dim.size_x[pidx_i]*transposed_dim.size_z[pidx_j], cudaMemcpyDeviceToHost));
        CUDA_CALL(cudaDeviceSynchronize());
    }

    for (size_t i = 0; i < comm_order2.size(); i++){
        size_t p_i = comm_order2[i];

        // Start non-blocking MPI recv
        MPI_Irecv(&recv_ptr[transposed_dim.start_x[p_i]], 1, MPI_RECV2[p_i], p_i, p_i, comm2, &recv_req[p_i]);

        size_t oslice = transposed_dim.size_x[pidx_i]*transposed_dim.size_z[pidx_j]*output_dim.start_y[p_i];

        if (i == 0)
            timer->stop_store("Second Transpose (First Send)");

        MPI_Isend(&send_ptr[oslice], sizeof(C_t)*transposed_dim.size_x[pidx_i]*transposed_dim.size_z[pidx_j]*output_dim.size_y[p_i],
            MPI_BYTE, p_i, pidx_i, comm2, &send_req[p_i]);
    }
}

template<typename T>
void MPIcuFFT_Pencil_Opt1<T>::Peer2Peer_Communication_SecondTranspose(void *complex_) {
    using C_t = typename cuFFT<T>::C_t;
    C_t *complex = cuFFT<T>::complex(complex_);
    C_t *recv_ptr, *temp_ptr; 

    temp_ptr = cuFFT<T>::complex(mem_d[0]);
    if (config.send_method2 == MPI_Type) {
        if (cuda_aware)
            recv_ptr = temp_ptr;
        else 
            recv_ptr = cuFFT<T>::complex(mem_h[0]);

        this->Peer2Peer_MPIType_SecondTranspose(complex_, (void *)recv_ptr);

        cudaMemcpy3DParms cpy_params = {0};
        cpy_params.srcPos = make_cudaPos(0, 0, output_dim.start_y[pidx_i]);
        cpy_params.srcPtr = make_cudaPitchedPtr(complex, sizeof(C_t)*transposed_dim.size_x[pidx_i], transposed_dim.size_x[pidx_i], transposed_dim.size_z[pidx_j]);
        cpy_params.dstPos = make_cudaPos(sizeof(C_t)*transposed_dim.start_x[pidx_i], 0, 0);
        cpy_params.dstPtr = make_cudaPitchedPtr(recv_ptr, sizeof(C_t)*global_size->Nx, global_size->Nx, output_dim.size_z[pidx_j]);
        cpy_params.extent = make_cudaExtent(sizeof(C_t)*transposed_dim.size_x[pidx_i], transposed_dim.size_z[pidx_j], output_dim.size_y[pidx_i]);
        cpy_params.kind   = cuda_aware?cudaMemcpyDeviceToDevice:cudaMemcpyDeviceToHost;

        CUDA_CALL(cudaMemcpy3DAsync(&cpy_params, streams[pidx_i])); 

        timer->stop_store("Transpose (Start Receive)");
        MPI_Waitall(partition->P1, recv_req.data(), MPI_STATUSES_IGNORE);

        if (!cuda_aware) {
            CUDA_CALL(cudaMemcpyAsync(temp_ptr, recv_ptr, 
                sizeof(C_t)*output_dim.size_x[0]*output_dim.size_y[pidx_i]*output_dim.size_z[pidx_j], cudaMemcpyHostToDevice));
        }
        CUDA_CALL(cudaDeviceSynchronize());
    } else {
        if (cuda_aware)
            recv_ptr = cuFFT<T>::complex(mem_d[1]);
        else
            recv_ptr = cuFFT<T>::complex(mem_h[0]);

        if (config.send_method2 == Sync)
            this->Peer2Peer_Sync_SecondTranspose(complex_, (void *)recv_ptr);
        else 
            this->Peer2Peer_Streams_SecondTranspose(complex_, (void *)recv_ptr);

        timer->stop_store("Second Transpose (Start Local Transpose)");
        // Transpose local block
        {
            cudaMemcpy3DParms cpy_params = {0};
            cpy_params.srcPos = make_cudaPos(0, 0, output_dim.start_y[pidx_i]);
            cpy_params.srcPtr = make_cudaPitchedPtr(complex, sizeof(C_t)*transposed_dim.size_x[pidx_i], transposed_dim.size_x[pidx_i], transposed_dim.size_z[pidx_j]);
            cpy_params.dstPos = make_cudaPos(sizeof(C_t)*transposed_dim.start_x[pidx_i], 0, 0);
            cpy_params.dstPtr = make_cudaPitchedPtr(temp_ptr, sizeof(C_t)*global_size->Nx, global_size->Nx, output_dim.size_z[pidx_j]);
            cpy_params.extent = make_cudaExtent(sizeof(C_t)*transposed_dim.size_x[pidx_i], transposed_dim.size_z[pidx_j], output_dim.size_y[pidx_i]);
            cpy_params.kind   = cudaMemcpyDeviceToDevice;

            CUDA_CALL(cudaMemcpy3DAsync(&cpy_params, streams[pidx_i]));            
        }

        timer->stop_store("Second Transpose (Start Receive)");
        // Start copying the received blocks to GPU memory (if !cuda_aware)
        // Otherwise the data is already correctly aligned. Therefore, we compute the last 1D FFT (x-direction) in the recv buffer (= temp1 buffer)
        int p;
        do {
            // recv_req contains one null handle (i.e. recv_req[pidx_i]) and P1-1 active handles
            // If all active handles are processed, Waitany will return MPI_UNDEFINED
            MPI_Waitany(partition->P1, recv_req.data(), &p, MPI_STATUSES_IGNORE);
            if (p == MPI_UNDEFINED)
                break;
            
            cudaMemcpy3DParms cpy_params = {0};
            cpy_params.srcPos = make_cudaPos(0, 0, 0);
            cpy_params.srcPtr = make_cudaPitchedPtr(&recv_ptr[transposed_dim.start_x[p]*transposed_dim.size_z[pidx_j]*output_dim.size_y[pidx_i]], 
                sizeof(C_t)*transposed_dim.size_x[p], transposed_dim.size_x[p], transposed_dim.size_z[pidx_j]);
            cpy_params.dstPos = make_cudaPos(sizeof(C_t)*transposed_dim.start_x[p], 0, 0);
            cpy_params.dstPtr = make_cudaPitchedPtr(temp_ptr, sizeof(C_t)*global_size->Nx, global_size->Nx, output_dim.size_z[pidx_j]);
            cpy_params.extent = make_cudaExtent(sizeof(C_t)*transposed_dim.size_x[p], transposed_dim.size_z[pidx_j], output_dim.size_y[pidx_i]);
            cpy_params.kind   = cuda_aware ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice;

            CUDA_CALL(cudaMemcpy3DAsync(&cpy_params, streams[(pidx_i + partition->P1 - p) % partition->P1]));   
        } while (p != MPI_UNDEFINED);

        // Wait for memcpy to complete
        CUDA_CALL(cudaDeviceSynchronize());
        timer->stop_store("Second Transpose (Finished Receive)");

    }
}

/* ***********************************************************************************************************************
*                                          Helper Methods for Second Global Transpose
*  *********************************************************************************************************************** *
*                                                           All2All
*  *********************************************************************************************************************** */

template<typename T>
void MPIcuFFT_Pencil_Opt1<T>::All2All_Sync_SecondTranspose(void *complex_) {
    printf("2. All2All Sync\n");
    using C_t = typename cuFFT<T>::C_t;
    C_t *complex = cuFFT<T>::complex(complex_);
    C_t *send_ptr, *recv_ptr, *temp_ptr;

    temp_ptr = cuFFT<T>::complex(mem_d[0]);
    if (cuda_aware) {
        send_ptr = complex;
        recv_ptr = cuFFT<T>::complex(mem_d[1]);
    } else {
        recv_ptr = cuFFT<T>::complex(mem_h[0]);
        send_ptr = cuFFT<T>::complex(mem_h[1]);
        CUDA_CALL(cudaMemcpyAsync(send_ptr, complex, 
            sizeof(C_t)*transposed_dim.size_y[0]*transposed_dim.size_x[pidx_i]*transposed_dim.size_z[pidx_j], cudaMemcpyDeviceToHost));
        CUDA_CALL(cudaDeviceSynchronize());
    }

    timer->stop_store("Transpose (Start All2All)");
    MPI_Alltoallv(send_ptr, sendcounts2.data(), sdispls2.data(), MPI_BYTE, 
                recv_ptr, recvcounts2.data(), rdispls2.data(), MPI_BYTE, comm2);
    timer->stop_store("Transpose (Finished All2All)");

    for (int p = 0; p < partition->P1; p++) {
        cudaMemcpy3DParms cpy_params = {0};
        cpy_params.srcPos = make_cudaPos(0, 0, 0);
        cpy_params.srcPtr = make_cudaPitchedPtr(&recv_ptr[transposed_dim.start_x[p]*transposed_dim.size_z[pidx_j]*output_dim.size_y[pidx_i]], 
            sizeof(C_t)*transposed_dim.size_x[p], transposed_dim.size_x[p], transposed_dim.size_z[pidx_j]);
        cpy_params.dstPos = make_cudaPos(sizeof(C_t)*transposed_dim.start_x[p], 0, 0);
        cpy_params.dstPtr = make_cudaPitchedPtr(temp_ptr, sizeof(C_t)*global_size->Nx, global_size->Nx, output_dim.size_z[pidx_j]);
        cpy_params.extent = make_cudaExtent(sizeof(C_t)*transposed_dim.size_x[p], transposed_dim.size_z[pidx_j], output_dim.size_y[pidx_i]);
        cpy_params.kind   = cuda_aware ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice;

        CUDA_CALL(cudaMemcpy3DAsync(&cpy_params, streams[p]));   
    }

    CUDA_CALL(cudaDeviceSynchronize());
}

template<typename T>
void MPIcuFFT_Pencil_Opt1<T>::All2All_MPIType_SecondTranspose(void *complex_) {
    printf("2. All2All MPI_Type\n");
    using C_t = typename cuFFT<T>::C_t;
    C_t *complex = cuFFT<T>::complex(complex_);
    C_t *send_ptr, *recv_ptr, *temp_ptr;

    temp_ptr = cuFFT<T>::complex(mem_d[0]);
    if (cuda_aware) {
        send_ptr = complex;
        recv_ptr = temp_ptr;
    } else {
        recv_ptr = cuFFT<T>::complex(mem_h[0]);
        send_ptr = cuFFT<T>::complex(mem_h[1]);
        CUDA_CALL(cudaMemcpyAsync(send_ptr, complex, 
            sizeof(C_t)*transposed_dim.size_y[0]*transposed_dim.size_x[pidx_i]*transposed_dim.size_z[pidx_j], cudaMemcpyDeviceToHost));
        CUDA_CALL(cudaDeviceSynchronize());
    }

    timer->stop_store("Transpose (Start All2All)");
    MPI_Alltoallw(send_ptr, sendcounts2.data(), sdispls2.data(), MPI_SND2.data(), 
                recv_ptr, recvcounts2.data(), rdispls2.data(), MPI_RECV2.data(), comm2);
    timer->stop_store("Transpose (Finished All2All)");

    if (!cuda_aware) {
        CUDA_CALL(cudaMemcpyAsync(temp_ptr, recv_ptr, 
            sizeof(C_t)*output_dim.size_x[0]*output_dim.size_y[pidx_i]*output_dim.size_z[pidx_j], cudaMemcpyHostToDevice));
        CUDA_CALL(cudaDeviceSynchronize());
    }
}

template<typename T>
void MPIcuFFT_Pencil_Opt1<T>::All2All_Communication_SecondTranspose(void *complex_) {
    if (config.send_method2 == MPI_Type) 
        this->All2All_MPIType_SecondTranspose(complex_);
    else 
        this->All2All_Sync_SecondTranspose(complex_);
}

template<typename T>
void MPIcuFFT_Pencil_Opt1<T>::execR2C(void *out, const void *in, int d) {
    if (!initialized)
        error("cuFFT plans are not yet initialized!");

    using R_t = typename cuFFT<T>::R_t;
    using C_t = typename cuFFT<T>::C_t;

    R_t *real    = cuFFT<T>::real(in);
    C_t *complex = cuFFT<T>::complex(out);

    if(fft3d) {
        CUFFT_CALL(cuFFT<T>::execR2C(planR2C, real, complex));
        CUDA_CALL(cudaDeviceSynchronize());
    } else {
        timer->start();
        // compute 1D FFT in z-direction
        // Afterwards the data alignment is [y][x][z] (input alignment [z][y][x])
        CUFFT_CALL(cuFFT<T>::execR2C(planR2C, real, complex));
        // Wait for 1D FFT in z-direction
        timer->stop_store("1D FFT Z-Direction");

        // Used to random_dist_1D test
        // TODO: Modifiy Testcases to handle transformed coordinate system
        if (d == 1) {
            CUDA_CALL(cudaDeviceSynchronize());
            return; 
        }

        C_t *recv_ptr, *send_ptr, *temp_ptr; 
  
        temp_ptr = cuFFT<T>::complex(mem_d[0]);
        if (cuda_aware) {
            recv_ptr = cuFFT<T>::complex(mem_d[1]);
        } else {
            recv_ptr = cuFFT<T>::complex(mem_h[0]);
            send_ptr = cuFFT<T>::complex(mem_h[1]);
        }

        /* ***********************************************************************************************************************
        *                                                       First Global Transpose
        *  *********************************************************************************************************************** */

        // Synchronize first 1D FFT (z-direction)
        CUDA_CALL(cudaDeviceSynchronize());

        
        if (config.comm_method == Peer2Peer)
            this->Peer2Peer_Communication_FirstTranspose((void *)complex);
        else 
            this->All2All_Communication_FirstTranspose((void *)complex);

        // avoid modification of complex, while MPI_Isendv is not done yet
        if (config.comm_method == Peer2Peer && cuda_aware)
            MPI_Waitall(partition->P2, send_req.data(), MPI_STATUSES_IGNORE);

        // Afterwards the data alignment is [x][z][y] (input alignment [z][y][x])
        CUFFT_CALL(cuFFT<T>::execC2C(planC2C_0, temp_ptr, complex, CUFFT_FORWARD));
        timer->stop_store("1D FFT Y-Direction");

        // used for random_dist_2D test
        // TODO: Modifiy Testcases to handle transformed coordinate system
        if (d == 2) {
            error("Option 1 does not support \"--fft-dim 2\".");
        } 

        /* ***********************************************************************************************************************
        *                                                     Second Global Transpose
        *  *********************************************************************************************************************** */

        /*** Hide preparation-phase by synchronizing later ***/

        // Synchronization and barrier of second 1D-FFT
        if (config.comm_method == Peer2Peer && !cuda_aware) {
            if (config.send_method == Streams)
                mpisend_thread1.join();
            MPI_Waitall(partition->P2, send_req.data(), MPI_STATUSES_IGNORE);
        }
        CUDA_CALL(cudaDeviceSynchronize());

        if (config.comm_method2 == Peer2Peer)
            this->Peer2Peer_Communication_SecondTranspose((void *)complex);
        else 
            this->All2All_Communication_SecondTranspose((void *)complex);

        // avoid modification of complex, while MPI_Isendv is not done yet
        if (config.comm_method2 == Peer2Peer && cuda_aware)
            MPI_Waitall(partition->P1, send_req.data(), MPI_STATUSES_IGNORE);

        // Compute the 1D FFT in x-direction
        CUFFT_CALL(cuFFT<T>::execC2C(planC2C_1, temp_ptr, complex, CUFFT_FORWARD));
        timer->stop_store("1D FFT X-Direction");
        if (config.comm_method2 == Peer2Peer && !cuda_aware) {
            if (config.send_method2 == Streams)
                mpisend_thread2.join();
            MPI_Waitall(partition->P1, send_req.data(), MPI_STATUSES_IGNORE);
        }
        CUDA_CALL(cudaDeviceSynchronize());
        timer->stop_store("Run complete");
    }
    cudaProfilerStop();
    if (config.warmup_rounds == 0) 
        timer->gather();
    else 
        config.warmup_rounds--;
}

template class MPIcuFFT_Pencil_Opt1<float>;
template class MPIcuFFT_Pencil_Opt1<double>;