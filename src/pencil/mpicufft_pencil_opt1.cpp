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

        // In the first global transpose, batches of P2 processes each communicate with each other
        if (!send_req.empty() || !recv_req.empty()){
            send_req.assign(std::min(send_req.size(), partition->P2), MPI_REQUEST_NULL);
            recv_req.assign(std::min(recv_req.size(), partition->P2), MPI_REQUEST_NULL);
        }
        send_req.resize(partition->P2, MPI_REQUEST_NULL);
        recv_req.resize(partition->P2, MPI_REQUEST_NULL);

        // Determine comm_order of first transpose
        this->commOrder_FirstTranspose();

        Callback_Params_Base base_params;
        std::vector<Callback_Params> params_array;

        if (!cuda_aware) {
            for (int i = 0; i < comm_order.size(); i++){
                size_t p_j = comm_order[i] % partition->P2;
                Callback_Params params = {&base_params, p_j};
                params_array.push_back(params);
            }
        }

        // Synchronize first 1D FFT (z-direction)
        CUDA_CALL(cudaDeviceSynchronize());

        // Start send thread
        std::thread mpisend_thread1;
        if (!cuda_aware)
            mpisend_thread1 = std::thread(&MPIcuFFT_Pencil_Opt1<T>::MPIsend_Thread_FirstCallback, this, std::ref(base_params), send_ptr);

        for (size_t i = 0; i < comm_order.size(); i++){
            size_t p_j = comm_order[i] % partition->P2;

            // Start non-blocking MPI recv
            MPI_Irecv(&recv_ptr[transposed_dim.size_x[pidx_i]*input_dim.start_y[p_j]*transposed_dim.size_z[pidx_j]],
                sizeof(C_t)*transposed_dim.size_x[pidx_i]*input_dim.size_y[p_j]*transposed_dim.size_z[pidx_j], MPI_BYTE,
                comm_order[i], p_j, comm, &recv_req[p_j]);

            size_t oslice = transposed_dim.start_z[p_j]*input_dim.size_y[pidx_j]*input_dim.size_x[pidx_i];
            if (!cuda_aware) {
                // TODO: Add option, where we have single memcpy before the loop
                // ~> Pro: Only single memcpy; Con: Additional Sync needed
                CUDA_CALL(cudaMemcpyAsync(&send_ptr[oslice], &complex[oslice],
                    sizeof(C_t)*transposed_dim.size_z[p_j]*input_dim.size_y[pidx_j]*input_dim.size_x[pidx_i], 
                    cudaMemcpyDeviceToHost, streams[p_j]));

                // After copy is complete, MPI starts a non-blocking send operation
                CUDA_CALL(cudaLaunchHostFunc(streams[p_j], this->MPIsend_Callback, (void *)&params_array[i]));
            } else {
                if (i == 0)
                    timer->stop_store("First Transpose (First Send)");

                MPI_Isend(&complex[oslice], 
                    sizeof(C_t)*transposed_dim.size_z[p_j]*input_dim.size_y[pidx_j]*input_dim.size_x[pidx_i], MPI_BYTE,
                    comm_order[i], pidx_j, comm, &send_req[p_j]);
            }            
        }

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
        // debug_h("\nRECV\n");
        int p;
        do {
            // recv_req contains one null handle (i.e. recv_req[pidx_j]) and P2-1 active handles
            // If all active handles are processed, Waitany will return MPI_UNDEFINED
            MPI_Waitany(partition->P2, recv_req.data(), &p, MPI_STATUSES_IGNORE);
            if (p == MPI_UNDEFINED)
                break;

            // debug_int("p", p);
            // debug_p("recv_ptr start", &recv_ptr[transposed_dim.size_x[pidx_i]*input_dim.start_y[p]*transposed_dim.size_z[pidx_j]]);
            // debug_int("recv_size", transposed_dim.size_x[pidx_i]*input_dim.size_y[p]*transposed_dim.size_z[pidx_j]);
            // debug_p("recv_ptr end", &recv_ptr[transposed_dim.size_x[pidx_i]*(input_dim.start_y[p]+input_dim.size_y[p])*transposed_dim.size_z[pidx_j]]);

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

        // avoid modification of complex, while MPI_Isendv is not done yet
        if (cuda_aware)
            MPI_Waitall(partition->P2, send_req.data(), MPI_STATUSES_IGNORE);

        // Afterwards the data alignment is [x][z][y] (input alignment [z][y][x])
        CUFFT_CALL(cuFFT<T>::execC2C(planC2C_0, temp_ptr, complex, CUFFT_FORWARD));
        timer->stop_store("1D FFT Y-Direction");

        // used for random_dist_2D test
        // TODO: Modifiy Testcases to handle transformed coordinate system
        if (d == 2) {
            CUDA_CALL(cudaDeviceSynchronize());
            if (!cuda_aware)
                mpisend_thread1.join();
            MPI_Waitall(partition->P2, send_req.data(), MPI_STATUSES_IGNORE);
            return;
        } 

        /* ***********************************************************************************************************************
        *                                                     Second Global Transpose
        *  *********************************************************************************************************************** */

        /*** Hide preparation-phase by synchronizing later ***/

        // Determine comm_order of second transpose
        this->commOrder_SecondTranspose();

        if (!cuda_aware) {
            params_array.clear();
            for (int i = 0; i < pcnt; i++){
                size_t p_i = (comm_order[i] - pidx_j) / partition->P2;
                Callback_Params params = {&base_params, p_i};
                params_array.push_back(params);
            }
        }

        // Synchronization and barrier of second 1D-FFT
        if (!cuda_aware) {
            mpisend_thread1.join();
            MPI_Waitall(partition->P2, send_req.data(), MPI_STATUSES_IGNORE);
        }
        CUDA_CALL(cudaDeviceSynchronize());

        // In the second global transpose, batches of P1 processes each communicate with each other
        if (!send_req.empty() || !recv_req.empty()){
            send_req.assign(std::min(send_req.size(), partition->P1), MPI_REQUEST_NULL);
            recv_req.assign(std::min(recv_req.size(), partition->P1), MPI_REQUEST_NULL);
        }
        send_req.resize(partition->P1, MPI_REQUEST_NULL);
        recv_req.resize(partition->P1, MPI_REQUEST_NULL);

        std::thread mpisend_thread2;
        if (!cuda_aware)
            mpisend_thread2 = std::thread(&MPIcuFFT_Pencil_Opt1<T>::MPIsend_Thread_SecondCallback, this, std::ref(base_params), send_ptr);
        timer->stop_store("Second Transpose (Preparation)");

        for (size_t i = 0; i < comm_order.size(); i++){
            size_t p_i = (comm_order[i] - pidx_j) / partition->P2;

            // Start non-blocking MPI recv
            MPI_Irecv(&recv_ptr[transposed_dim.start_x[p_i]*output_dim.size_y[pidx_i]*output_dim.size_z[pidx_j]], 
                sizeof(C_t)*transposed_dim.size_x[p_i]*output_dim.size_y[pidx_i]*output_dim.size_z[pidx_j], MPI_BYTE,
                comm_order[i], p_i, comm, &recv_req[p_i]);

            size_t oslice = transposed_dim.size_x[pidx_i]*transposed_dim.size_z[pidx_j]*output_dim.start_y[p_i];
            if (!cuda_aware) {
                CUDA_CALL(cudaMemcpyAsync(&send_ptr[oslice], &complex[oslice], 
                    sizeof(C_t)*transposed_dim.size_x[pidx_i]*transposed_dim.size_z[pidx_j]*output_dim.size_y[p_i], 
                    cudaMemcpyDeviceToHost, streams[p_i]));

                // After copy is complete, MPI starts a non-blocking send operation
                CUDA_CALL(cudaLaunchHostFunc(streams[p_i], this->MPIsend_Callback, (void *)&params_array[i]));
            } else {
                if (i == 0)
                    timer->stop_store("Second Transpose (First Send)");

                MPI_Isend(&complex[oslice], sizeof(C_t)*transposed_dim.size_x[pidx_i]*transposed_dim.size_z[pidx_j]*output_dim.size_y[p_i],
                    MPI_BYTE, comm_order[i], pidx_i, comm, &send_req[p_i]);
            }
        }
        
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

        // avoid modification of complex, while MPI_Isendv is not done yet
        if (cuda_aware)
            MPI_Waitall(partition->P1, send_req.data(), MPI_STATUSES_IGNORE);

        // Compute the 1D FFT in x-direction
        CUFFT_CALL(cuFFT<T>::execC2C(planC2C_1, temp_ptr, complex, CUFFT_FORWARD));
        timer->stop_store("1D FFT X-Direction");
        if (!cuda_aware) {
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