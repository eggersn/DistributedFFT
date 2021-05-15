#include "mpicufft_pencil.hpp"
#include "cufft.hpp"
#include <cuda_runtime.h>
#include <charconv>

#define error(e) {                  \
    throw std::runtime_error(e);    \
}

#define CUDA_CALL(x) do { if((x)!=cudaSuccess) {        \
    printf("Error %d at %s:%d\n",x,__FILE__,__LINE__);  \
    exit(EXIT_FAILURE); }} while(0)

#define CUFFT_CALL(x) do { if((x)!=CUFFT_SUCCESS) {     \
    printf("Error %d at %s:%d\n",x,__FILE__,__LINE__);  \
    exit(EXIT_FAILURE); }} while(0)

#define DEBUG 0
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

#define sdebug(d, v, pidx_i, pidx_j) {                                                 \
  if (DEBUG == 1) {                                                   \
    printf("DEBUG (%d,%d): %s: %s in %s:%d\n",pidx_i,pidx_j,d,v,__FILE__,__LINE__); \
  }                                                                   \
}

#define sdebug_h(v, pidx_i, pidx_j) {                                                  \
  if (DEBUG == 1) {                                                   \
    printf("%s",v);                \
  }                                                                   \
}

#define sdebug_int(d, v, pidx_i, pidx_j) {                                             \
  if (DEBUG == 1) {                                                   \
    printf("DEBUG (%d,%d): %s: %d in %s:%d\n",pidx_i,pidx_j,d,v,__FILE__,__LINE__); \
  }                                                                   \
}

#define sdebug_p(d, v, pidx_i, pidx_j) {                                             \
  if (DEBUG == 1) {                                                   \
    printf("DEBUG (%d,%d): %s: %p in %s:%d\n",pidx_i,pidx_j,d,v,__FILE__,__LINE__); \
  }                                                                   \
}

template<typename T>
MPIcuFFT_Pencil<T>::MPIcuFFT_Pencil(MPI_Comm comm, bool mpi_cuda_aware, int max_world_size) : MPIcuFFT<T>(comm, mpi_cuda_aware, max_world_size) {
    pidx_i = 0;
    pidx_j = 0;

    ws_c2c_0 = 0;

    planR2C = 0;
    planC2C_1 = 0;
}

template<typename T> 
MPIcuFFT_Pencil<T>::~MPIcuFFT_Pencil() {
    if (planR2C) 
        CUFFT_CALL(cufftDestroy(planR2C));
    if (!planC2C_0.empty()){
        for (auto &handle : planC2C_0)
            CUFFT_CALL(cufftDestroy(handle));
    }
    if (planC2C_1) 
        CUFFT_CALL(cufftDestroy(planC2C_1));
}

template<typename T>
void MPIcuFFT_Pencil<T>::commOrder_FirstTranspose() {
    comm_order.clear();
    for (int j = 1; j < partition->P2; j++){
        comm_order.push_back(pidx_i*partition->P2 + (pidx_j+j)%partition->P2);
        debug_int("1. comm_order", comm_order[j-1]);
    }
}

template<typename T>
void MPIcuFFT_Pencil<T>::commOrder_SecondTranspose() {
    comm_order.clear();
    for (int i = 1; i < partition->P1; i++){
        comm_order.push_back(((pidx_i+i)%partition->P1)*partition->P2 + pidx_j);
        debug_int("2. comm_order", comm_order[i-1]);
    }
}

template<typename T>
void MPIcuFFT_Pencil<T>::initFFT(GlobalSize *global_size_, Partition *partition_, bool allocate) {
    global_size = global_size_;
    partition = partition_;

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
    debug_h("\ninput_dim\n");
    debug_int("input_dim.size_x[pidx_i]", input_dim.size_x[pidx_i]);
    debug_int("input_dim.size_y[pidx_j]", input_dim.size_y[pidx_j]);
    debug_int("input_dim.size_z[0]", input_dim.size_z[0]);
    // transposed_dim:
    transposed_dim.size_x = input_dim.size_x;
    transposed_dim.size_y.resize(1, global_size->Ny);
    transposed_dim.size_z.resize(partition->P2, (global_size->Nz/2+1)/partition->P2);
    for (int k = 0; k < (global_size->Nz/2+1)%partition->P2; k++)
        transposed_dim.size_z[k]++;
    transposed_dim.computeOffsets();
    debug_h("\ntransposed_dim\n");
    debug_int("transposed_dim.size_x[pidx_i]", transposed_dim.size_x[pidx_i]);
    debug_int("transposed_dim.size_y[0]", transposed_dim.size_y[0]);
    debug_int("transposed_dim.size_z[pidx_j]", transposed_dim.size_z[pidx_j]);
    // output_dim:
    output_dim.size_x.resize(1, global_size->Nx);
    output_dim.size_y.resize(partition->P1, global_size->Ny/partition->P1);
    for (int j = 0; j < global_size->Ny%partition->P1; j++)
        output_dim.size_y[j]++;
    output_dim.size_z = transposed_dim.size_z;
    output_dim.computeOffsets();
    debug_h("\noutput_dim\n");
    debug_int("output_dim.size_x[0]", output_dim.size_x[0]);
    debug_int("output_dim.size_y[pidx_i]", output_dim.size_y[pidx_i]);
    debug_int("output_dim.size_z[pidx_j]", output_dim.size_z[pidx_j]);

    // Determine comm_order of first transpose
    this->commOrder_FirstTranspose();

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
        size_t ws_c2c_1;
        size_t batch[3] = {input_dim.size_x[pidx_i]*input_dim.size_y[pidx_j], 0, output_dim.size_y[pidx_i]*output_dim.size_z[pidx_j]};

        CUFFT_CALL(cufftCreate(&planC2C_1));      
        CUFFT_CALL(cufftSetAutoAllocation(planC2C_1, 0));

        long long int n[3] = {static_cast<long long int>(output_dim.size_x[0]), static_cast<long long int>(transposed_dim.size_y[0]), 
            static_cast<long long int>(input_dim.size_z[0])};

        // (1) For the first 1D FFT (z-direction), we can use the default data layout (therefore the nullptr; see cuFFT documentation for details)
        CUFFT_CALL(cufftMakePlanMany64(planR2C, 1, &n[2], //plan, rank, *n
            nullptr, 0, 0, //*inembed, istride, idist
            nullptr, 0, 0, //*onembed, ostride, odist
            cuFFT<T>::R2Ctype, static_cast<long long int>(batch[0]), &ws_r2c)); //type, batch, worksize

        // (2) For the second 1D FFT (y-direction), we need to either reuse the same plan or create the identical plan multiple times 
        // while associating each one with a different stream
        num_of_streams = std::min(transposed_dim.size_x[pidx_i], transposed_dim.size_z[pidx_j]);
        batch[1] = std::max(transposed_dim.size_x[pidx_i], transposed_dim.size_z[pidx_j]);
        long long int nembed[1] = {(transposed_dim.size_x[pidx_i] <= transposed_dim.size_z[pidx_j]) ? 
            1 : static_cast<long long int>(transposed_dim.size_z[pidx_j] * transposed_dim.size_y[0])};

        debug_h("\nCUFFT Make Plan (y-direction)\n");
        debug_int("nembed", nembed[0]);
        debug_int("num_of_streams", num_of_streams);
        debug_int("batch", batch[1]);
        debug_int("istride/ostride", transposed_dim.size_z[pidx_j]);

        for (int s = 0; s < num_of_streams; s++){
            // Create multiple plans and associate each one with a different stream
            cudaStream_t stream;
            CUDA_CALL(cudaStreamCreate(&stream));
            streams.push_back(stream);
            
            cufftHandle handle;
            CUFFT_CALL(cufftCreate(&handle));      
            CUFFT_CALL(cufftSetAutoAllocation(handle, 0));

            CUFFT_CALL(cufftMakePlanMany64(handle, 1, &n[1], //plan, rank, *n
                nembed, static_cast<long long int>(transposed_dim.size_z[pidx_j]), nembed[0], //*inembed, istride, idist
                nembed, static_cast<long long int>(transposed_dim.size_z[pidx_j]), nembed[0], //*onembed, ostride, odist
                cuFFT<T>::C2Ctype, static_cast<long long int>(batch[1]), &ws_c2c_0)); //type, batch, worksize
            CUFFT_CALL(cufftSetStream(handle, stream));
            planC2C_0.push_back(handle);
        }

        // (3) For the third 1D FFT (x-direction) the distance between two consecutive signals in a batch is 1.
        // Therefore, a single plan is sufficient.
        nembed[0] = 1;

        debug_h("\nCUFFT Make Plan (x-direction)\n");
        debug_int("nembed", nembed[0]);
        debug_int("batch", batch[2]);
        debug_int("output_dim.size_y[pidx_i]", output_dim.size_y[pidx_i]);
        debug_int("output_dim.size_z[pidx_j]", output_dim.size_z[pidx_j]);
        debug_int("istride/ostride", output_dim.size_y[pidx_i]*output_dim.size_z[pidx_j]);

        CUFFT_CALL(cufftMakePlanMany64(planC2C_1, 1, &n[0], //plan, rank, *n
            nembed, static_cast<long long int>(output_dim.size_y[pidx_i]*output_dim.size_z[pidx_j]), 1, //*inembed, istride, idist
            nembed, static_cast<long long int>(output_dim.size_y[pidx_i]*output_dim.size_z[pidx_j]), 1, //*onembed, ostride, odist
            cuFFT<T>::C2Ctype, static_cast<long long int>(batch[2]), &ws_c2c_1)); //type, batch, worksize

        fft_worksize = std::max(ws_r2c, num_of_streams*ws_c2c_0);
        fft_worksize = std::max(fft_worksize, ws_c2c_1);        
    }

    // Max space for the process's pencil. Needed for send/recv buffer
    domainsize = std::max(input_dim.size_x[pidx_i]*input_dim.size_y[pidx_j]*(input_dim.size_z[0]/2+1), 
        transposed_dim.size_x[pidx_i]*transposed_dim.size_y[0]*transposed_dim.size_z[pidx_j]);
    domainsize = sizeof(C_t) * std::max(domainsize, output_dim.size_x[0]*output_dim.size_y[pidx_i]*output_dim.size_z[pidx_j]);

    if (fft_worksize < domainsize)
        fft_worksize = domainsize;

    // worksize_d is split into 4 parts (2 if not cuda aware):
    // 1. temp buffer, for local transpose (if cuda aware) or buffer for host->device transfer of recv buffer
    // 2. recv buffer (if cuda aware)
    // 3. send buffer (if cuda aware)
    // 4. actual workspace (slot 2 if not cuda aware)
    worksize_d = fft_worksize + (fft3d ? 0 : (cuda_aware ? 3*domainsize : domainsize ));
    // if not cuda aware, then recv and send buffer are on the host side
    worksize_h = (cuda_aware || fft3d ? 0 : 2*domainsize);

    debug_int("worksize_d", worksize_d);
    debug_int("worksize_h", worksize_h);
    debug_int("domainsize", domainsize);

    if (allocate)
        this->setWorkArea();

    CUDA_CALL(cudaDeviceSynchronize());
}

template<typename T>
void MPIcuFFT_Pencil<T>::setWorkArea(void *device, void *host){
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

    // mem_d stores pointer to all (above described) workspaces (temp, recv, send, actual)
    mem_d.clear();
    for (size_t i = 0; i < 1 + (fft3d ? 0 : (cuda_aware ? 3 : 1)); i++)
        mem_d.push_back(&static_cast<char*>(workarea_d)[i*domainsize]);

    debug_int("mem_d size", mem_d.size());
    
    if (fft3d) {
        CUFFT_CALL(cufftSetWorkArea(planR2C, mem_d[0]));
    } else if (!fft3d) {
        // The second 1D FFT (y-direction) is split into multiple streams,
        // therefore each one has its own workspace
        size_t offset = cuda_aware ? 3 : 1;
        for (size_t s = 1; s < num_of_streams; s++)
            mem_d.push_back(&static_cast<char*>(workarea_d)[offset*domainsize+s*ws_c2c_0]);

        debug_int("mem_d size", mem_d.size());

        // Since each computation R2C, C2C_0 and C2C_1 is split by a global transpose,
        // the same workspace can be reused
        CUFFT_CALL(cufftSetWorkArea(planR2C, mem_d[offset]));
        for (int i = 0; i < planC2C_0.size(); i++)
            CUFFT_CALL(cufftSetWorkArea(planC2C_0[i], mem_d[offset+i]));
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

template<typename S, typename T>
struct Callback_Params {
    size_t i;
    T *send_ptr;
    MPIcuFFT_Pencil<S> *mpicufft_pencil;
};

template<typename T>
void MPIcuFFT_Pencil<T>::MPIsend_Callback_FirstTranspose(void *data) {
    using C_t = typename cuFFT<T>::C_t;

    Callback_Params<T, C_t> *params = (Callback_Params<T, C_t>*)data;
    size_t i = params->i;
    C_t *send_ptr = params->send_ptr;
    MPIcuFFT_Pencil<T> *mpicufft_pencil = params->mpicufft_pencil;

    size_t p_j = mpicufft_pencil->comm_order[i] % mpicufft_pencil->partition->P2;

    sdebug_h("\nCallback First Transpose", mpicufft_pencil->pidx_i, mpicufft_pencil->pidx_j);
    sdebug_int("i", i, mpicufft_pencil->pidx_i, mpicufft_pencil->pidx_j);
    sdebug_int("pidx_i", mpicufft_pencil->pidx_i, mpicufft_pencil->pidx_i, mpicufft_pencil->pidx_j);
    sdebug_int("pidx_j", mpicufft_pencil->pidx_j, mpicufft_pencil->pidx_i, mpicufft_pencil->pidx_j);
    sdebug_int("p_j", p_j, mpicufft_pencil->pidx_i, mpicufft_pencil->pidx_j);
    sdebug_int("input_dim.size_x[pidx_i]", mpicufft_pencil->input_dim.size_x[mpicufft_pencil->pidx_i], mpicufft_pencil->pidx_i, mpicufft_pencil->pidx_j);
    sdebug_int("input_dim.size_y[pidx_j]", mpicufft_pencil->input_dim.size_y[mpicufft_pencil->pidx_j], mpicufft_pencil->pidx_i, mpicufft_pencil->pidx_j);
    sdebug_int("transposed_dim.start_z[p_j]", mpicufft_pencil->transposed_dim.start_z[p_j], mpicufft_pencil->pidx_i, mpicufft_pencil->pidx_j);
    sdebug_int("transposed_dim.size_z[p_j]", mpicufft_pencil->transposed_dim.size_z[p_j], mpicufft_pencil->pidx_i, mpicufft_pencil->pidx_j);
    sdebug_int("comm_order[i]", mpicufft_pencil->comm_order[i], mpicufft_pencil->pidx_i, mpicufft_pencil->pidx_j);
    sdebug_int("send_ptr offset", mpicufft_pencil->input_dim.size_x[mpicufft_pencil->pidx_i]*mpicufft_pencil->input_dim.size_y[mpicufft_pencil->pidx_j]*mpicufft_pencil->transposed_dim.start_z[p_j], mpicufft_pencil->pidx_i, mpicufft_pencil->pidx_j);
    sdebug_int("size", sizeof(C_t)*mpicufft_pencil->input_dim.size_x[mpicufft_pencil->pidx_i]*mpicufft_pencil->input_dim.size_y[mpicufft_pencil->pidx_j]*mpicufft_pencil->transposed_dim.size_z[p_j], mpicufft_pencil->pidx_i, mpicufft_pencil->pidx_j);
    sdebug_p("send_ptr", &send_ptr[mpicufft_pencil->input_dim.size_x[mpicufft_pencil->pidx_i]*mpicufft_pencil->input_dim.size_y[mpicufft_pencil->pidx_j]*mpicufft_pencil->transposed_dim.start_z[p_j]], mpicufft_pencil->pidx_i, mpicufft_pencil->pidx_j);
    sdebug_h("\n", mpicufft_pencil->pidx_i, mpicufft_pencil->pidx_j);

    MPI_Isend(&send_ptr[mpicufft_pencil->input_dim.size_x[mpicufft_pencil->pidx_i]*mpicufft_pencil->input_dim.size_y[mpicufft_pencil->pidx_j]*mpicufft_pencil->transposed_dim.start_z[p_j]],
    sizeof(C_t)*mpicufft_pencil->input_dim.size_x[mpicufft_pencil->pidx_i]*mpicufft_pencil->input_dim.size_y[mpicufft_pencil->pidx_j]*mpicufft_pencil->transposed_dim.size_z[p_j], MPI_BYTE,
    mpicufft_pencil->comm_order[i], mpicufft_pencil->comm_order[i], mpicufft_pencil->comm, &(mpicufft_pencil->send_req[p_j]));
}

template<typename T>
void MPIcuFFT_Pencil<T>::MPIsend_Callback_SecondTranspose(void *data) {
    using C_t = typename cuFFT<T>::C_t;

    Callback_Params<T, C_t> *params = (Callback_Params<T, C_t>*)data;
    size_t i = params->i;
    C_t *send_ptr = params->send_ptr;
    MPIcuFFT_Pencil<T> *mpicufft_pencil = params->mpicufft_pencil;

    size_t p_i = (mpicufft_pencil->comm_order[i] - mpicufft_pencil->pidx_j) / mpicufft_pencil->partition->P2;

    MPI_Isend(&send_ptr[mpicufft_pencil->transposed_dim.size_x[mpicufft_pencil->pidx_i]*mpicufft_pencil->transposed_dim.size_z[mpicufft_pencil->pidx_j]*mpicufft_pencil->output_dim.start_y[p_i]], 
        sizeof(C_t)*mpicufft_pencil->transposed_dim.size_x[mpicufft_pencil->pidx_i]*mpicufft_pencil->output_dim.size_y[p_i]*mpicufft_pencil->transposed_dim.size_z[mpicufft_pencil->pidx_j], MPI_BYTE,
        mpicufft_pencil->comm_order[i], mpicufft_pencil->comm_order[i], mpicufft_pencil->comm, &(mpicufft_pencil->send_req[p_i]));
}

template<typename T>
void MPIcuFFT_Pencil<T>::execR2C(void *out, const void *in) {
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
        // compute 1D FFT in z-direction
        CUFFT_CALL(cuFFT<T>::execR2C(planR2C, real, complex));

        // Used to random_dist_1D test
        // CUDA_CALL(cudaDeviceSynchronize());
        // return; 

        C_t *recv_ptr, *send_ptr, *temp_ptr; 
  
        temp_ptr = cuFFT<T>::complex(mem_d[0]);
        if (cuda_aware) {
            recv_ptr = cuFFT<T>::complex(mem_d[1]);
            send_ptr = cuFFT<T>::complex(mem_d[2]);
        } else {
            recv_ptr = cuFFT<T>::complex(mem_h[0]);
            send_ptr = cuFFT<T>::complex(mem_h[1]);
        }

        /* ***********************************************************************************************************************
        *                                                       First Global Transpose
        *  *********************************************************************************************************************** */

        // Verify that there are at least P2 streams
        if (streams.size() < partition->P2) {
            for (int i = streams.size(); i < partition->P2; i++){
                cudaStream_t stream;
                CUDA_CALL(cudaStreamCreate(&stream));
                streams.push_back(stream);
            }
        }

        // In the first global transpose, batches of P2 processes each communicate with each other
        if (!send_req.empty() || !recv_req.empty()){
            send_req.assign(std::min(send_req.size(), partition->P2), MPI_REQUEST_NULL);
            recv_req.assign(std::min(recv_req.size(), partition->P2), MPI_REQUEST_NULL);
        }
        send_req.resize(partition->P2, MPI_REQUEST_NULL);
        recv_req.resize(partition->P2, MPI_REQUEST_NULL);

        // Wait for 1D FFT in z-direction
        CUDA_CALL(cudaDeviceSynchronize());

        for (size_t i = 0; i < comm_order.size(); i++){
            size_t p_i = pidx_i;
            size_t p_j = comm_order[i] % partition->P2;

            // Start non-blocking MPI recv
            MPI_Irecv(&recv_ptr[transposed_dim.size_x[pidx_i]*input_dim.start_y[p_j]*transposed_dim.size_z[pidx_j]],
                sizeof(C_t)*transposed_dim.size_x[pidx_i]*input_dim.size_y[p_j]*transposed_dim.size_z[pidx_j], MPI_BYTE,
                comm_order[i], p_j, comm, &recv_req[p_j]);

            // Copy 1D FFT results (z-direction) to the send buffer
            // cudaPos = {z (bytes), y (elements), x (elements)}
            // cudaPitchedPtr = {pointer, pitch (byte), allocation width, allocation height}
            // cudaExtend = {width, height, depth}
            cudaMemcpy3DParms cpy_params = {0};
            cpy_params.srcPos = make_cudaPos(transposed_dim.start_z[p_j]*sizeof(C_t), 0, 0);
            cpy_params.srcPtr = make_cudaPitchedPtr(complex, global_size->Nz_out*sizeof(C_t), global_size->Nz_out, input_dim.size_y[pidx_j]);
            cpy_params.dstPos = make_cudaPos(0,0,0); // offset cannot be specified by cuda position allow ~> use pointer instead
            cpy_params.dstPtr = make_cudaPitchedPtr(&send_ptr[input_dim.size_x[pidx_i]*input_dim.size_y[pidx_j]*transposed_dim.start_z[p_j]],
                transposed_dim.size_z[p_j]*sizeof(C_t), transposed_dim.size_z[p_j], input_dim.size_y[pidx_j]);
            cpy_params.extent = make_cudaExtent(transposed_dim.size_z[p_j]*sizeof(C_t), input_dim.size_y[pidx_j], input_dim.size_x[pidx_i]);
            cpy_params.kind   = cuda_aware ? cudaMemcpyDeviceToDevice : cudaMemcpyDeviceToHost;

            CUDA_CALL(cudaMemcpy3DAsync(&cpy_params, streams[p_j]));
            CUDA_CALL(cudaDeviceSynchronize());

            // // After copy is complete, MPI starts a non-blocking send operation
            // Callback_Params<T, C_t> params = {i, send_ptr, this};
            // CUDA_CALL(cudaLaunchHostFunc(streams[p_j], this->MPIsend_Callback_FirstTranspose, (void *)&params));

            debug_h("\nCallback First Transpose\n");
            debug_int("i", i);
            debug_int("pidx_i", pidx_i);
            debug_int("pidx_j", pidx_j);
            debug_int("p_j", p_j);
            debug_int("input_dim.size_x[pidx_i]", input_dim.size_x[pidx_i]);
            debug_int("input_dim.size_y[pidx_j]", input_dim.size_y[pidx_j]);
            debug_int("transposed_dim.start_z[p_j]", transposed_dim.start_z[p_j]);
            debug_int("transposed_dim.size_z[p_j]", transposed_dim.size_z[p_j]);
            debug_int("comm_order[i]", comm_order[i]);
            debug_int("send_ptr offset", input_dim.size_x[pidx_i]*input_dim.size_y[pidx_j]*transposed_dim.start_z[p_j]);
            debug_int("size", sizeof(C_t)*input_dim.size_x[pidx_i]*input_dim.size_y[pidx_j]*transposed_dim.size_z[p_j]);
            debug_p("send_ptr", &send_ptr[input_dim.size_x[pidx_i]*input_dim.size_y[pidx_j]*transposed_dim.start_z[p_j]]);
            debug_h("\n");

            MPI_Isend(&send_ptr[input_dim.size_x[pidx_i]*input_dim.size_y[pidx_j]*transposed_dim.start_z[p_j]],
                sizeof(C_t)*input_dim.size_x[pidx_i]*input_dim.size_y[pidx_j]*transposed_dim.size_z[p_j], MPI_BYTE,
                comm_order[i], pidx_j, comm, &(send_req[p_j]));
        }

        // Transpose local block and copy it to the temp buffer
        // This is required, since otherwise the data layout is not consecutive
        {
            cudaMemcpy3DParms cpy_params = {0};
            cpy_params.srcPos = make_cudaPos(transposed_dim.start_z[pidx_j]*sizeof(C_t), 0, 0);
            cpy_params.srcPtr = make_cudaPitchedPtr(complex, global_size->Nz_out*sizeof(C_t), global_size->Nz_out, input_dim.size_y[pidx_j]);
            cpy_params.dstPos = make_cudaPos(0, input_dim.start_y[pidx_j], 0);
            cpy_params.dstPtr = make_cudaPitchedPtr(temp_ptr, transposed_dim.size_z[pidx_j]*sizeof(C_t), transposed_dim.size_z[pidx_j], transposed_dim.size_y[0]);
            cpy_params.extent = make_cudaExtent(transposed_dim.size_z[pidx_j]*sizeof(C_t), input_dim.size_y[pidx_j], input_dim.size_x[pidx_i]);
            cpy_params.kind   = cudaMemcpyDeviceToDevice;

            CUDA_CALL(cudaMemcpy3DAsync(&cpy_params, streams[pidx_j]));
        }

        // Start copying the received blocks to the temp buffer, where the second 1D FFT (y-direction) can be computed
        // Since the received data has to be realigned (independent of cuda_aware), we use cudaMemcpy3D.
        int p;
        do {
            // recv_req contains one null handle (i.e. recv_req[pidx_j]) and P2-1 active handles
            // If all active handles are processed, Waitany will return MPI_UNDEFINED
            MPI_Waitany(partition->P2, recv_req.data(), &p, MPI_STATUSES_IGNORE);
            if (p == MPI_UNDEFINED)
                break;

            debug_h("\nFirst Global Receive\n");
            debug_int("p", p);
            debug_int("input_dim.size_y[p]", input_dim.size_y[p]);
            debug_int("transposed_dim.size_x[pidx_i]", transposed_dim.size_x[pidx_i]);
            debug_int("transposed_dim.size_z[pidx_j]", transposed_dim.size_z[pidx_j]);
            debug_int("transposed_dim.start_y[p]", transposed_dim.start_y[p]);
            debug_int("recv_ptr offset", transposed_dim.size_x[pidx_i]*transposed_dim.size_z[pidx_j]*transposed_dim.start_y[p]);
            debug_int("size", transposed_dim.size_z[pidx_j]*sizeof(C_t)*input_dim.size_y[p]*input_dim.size_x[pidx_i]);
            debug_p("send_ptr",&recv_ptr[transposed_dim.size_x[pidx_i]*transposed_dim.size_z[pidx_j]*transposed_dim.start_y[p]]);
            debug_h("\n");

            // At this point, we received data of one of the P2-1 other relevant processes
            cudaMemcpy3DParms cpy_params = {0};
            cpy_params.srcPos = make_cudaPos(0, 0, 0);
            cpy_params.srcPtr = make_cudaPitchedPtr(&recv_ptr[transposed_dim.size_x[pidx_i]*input_dim.start_y[p]*transposed_dim.size_z[pidx_j]],
                transposed_dim.size_z[pidx_j]*sizeof(C_t), transposed_dim.size_z[pidx_j], input_dim.size_y[p]);
            cpy_params.dstPos = make_cudaPos(0, input_dim.start_y[p], 0);
            cpy_params.dstPtr = make_cudaPitchedPtr(temp_ptr, transposed_dim.size_z[pidx_j]*sizeof(C_t), transposed_dim.size_z[pidx_j], transposed_dim.size_y[0]);
            cpy_params.extent = make_cudaExtent(transposed_dim.size_z[pidx_j]*sizeof(C_t), input_dim.size_y[p], input_dim.size_x[pidx_i]);
            cpy_params.kind   = cuda_aware ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice;

            CUDA_CALL(cudaMemcpy3DAsync(&cpy_params, streams[(pidx_j + partition->P2 - p) % partition->P2]));   // TODO: check if this is the best stream for selection!
        } while (p != MPI_UNDEFINED);

        // For the 1D FFT in y-direction, all data packages have to be received
        CUDA_CALL(cudaDeviceSynchronize());

        // If transposed_dim.size_x[pidx_i] <= transposed_dim.size_z[pidx_j] then:
        // We compute multiple 1D FFT batches, where each batch is a slice in y-z direction
        // Otherwise: We compute multiple 1D FFT batches, where each batch is a slice in x-y direction
        // ~> Goal: Make the batches as large as possible
        size_t batches_offset = 1;
        if (transposed_dim.size_x[pidx_i] <= transposed_dim.size_z[pidx_j])
            batches_offset = transposed_dim.size_z[pidx_j] * transposed_dim.size_y[0];

        debug_h("\n1D FFT in y Direction\n");
        debug_int("batches_offset", batches_offset);
        debug_int("num_of_streams", num_of_streams);

        for (size_t i = 0; i < num_of_streams; i++){
            // Compute the batch of 1D FFT's in y-direction
            CUFFT_CALL(cuFFT<T>::execC2C(planC2C_0[i], &temp_ptr[i*batches_offset], &complex[i*batches_offset], CUFFT_FORWARD));
        }
        CUDA_CALL(cudaDeviceSynchronize());
        MPI_Waitall(partition->P2, send_req.data(), MPI_STATUSES_IGNORE);

        // used for random_dist_2D test
        // return;

        /* ***********************************************************************************************************************
        *                                                     Second Global Transpose
        *  *********************************************************************************************************************** */

        // Determine comm_order of second transpose
        this->commOrder_SecondTranspose();

        // Verify that there are at least P1 streams
        if (streams.size() < partition->P1) {
            for (int i = streams.size(); i < partition->P1; i++){
                cudaStream_t stream;
                CUDA_CALL(cudaStreamCreate(&stream));
                streams.push_back(stream);
            }
        }

        // In the second global transpose, batches of P1 processes each communicate with each other
        if (!send_req.empty() || !recv_req.empty()){
            send_req.assign(std::min(send_req.size(), partition->P1), MPI_REQUEST_NULL);
            recv_req.assign(std::min(recv_req.size(), partition->P1), MPI_REQUEST_NULL);
        }
        send_req.resize(partition->P1, MPI_REQUEST_NULL);
        recv_req.resize(partition->P1, MPI_REQUEST_NULL);

        for (size_t i = 0; i < comm_order.size(); i++){
            size_t p_j = pidx_j;
            size_t p_i = (comm_order[i] - p_j) / partition->P2;

            // Start non-blocking MPI recv
            MPI_Irecv(&recv_ptr[transposed_dim.start_x[p_i]*output_dim.size_y[pidx_i]*output_dim.size_z[p_j]], 
                sizeof(C_t)*transposed_dim.size_x[p_i]*output_dim.size_y[pidx_i]*output_dim.size_z[p_j], MPI_BYTE,
                comm_order[i], p_i, comm, &recv_req[p_i]);

            // Copy 1D FFT results (y-direction) to the send buffer
            // cudaPos = {z (bytes), y (elements), x (elements)}
            // cudaPitchedPtr = {pitch (byte), pointer, allocation width, allocation height}
            // cudaExtend = {depth, height, width}
            cudaMemcpy3DParms cpy_params = {0};
            cpy_params.srcPos = make_cudaPos(0, output_dim.start_y[p_i], 0);
            cpy_params.srcPtr = make_cudaPitchedPtr(complex, sizeof(C_t)*transposed_dim.size_z[pidx_j], transposed_dim.size_z[pidx_j], transposed_dim.size_y[0]);
            cpy_params.dstPos = make_cudaPos(0,0,0); // offset cannot be specified by cuda position allow ~> use pointer instead
            cpy_params.dstPtr = make_cudaPitchedPtr(&send_ptr[transposed_dim.size_x[pidx_i]*transposed_dim.size_z[pidx_j]*output_dim.start_y[p_i]],
                sizeof(C_t)*transposed_dim.size_z[pidx_j], transposed_dim.size_z[pidx_j], output_dim.size_y[p_i]);
            cpy_params.extent = make_cudaExtent(sizeof(C_t)*transposed_dim.size_z[pidx_j], output_dim.size_y[p_i], transposed_dim.size_x[pidx_i]);
            cpy_params.kind   = cuda_aware ? cudaMemcpyDeviceToDevice : cudaMemcpyDeviceToHost;

            CUDA_CALL(cudaMemcpy3DAsync(&cpy_params, streams[p_i]));
            CUDA_CALL(cudaDeviceSynchronize());

            debug_h("\nCallback Second Transpose\n");
            debug_int("i", i);
            debug_int("p_i", p_i);
            debug_int("comm_order[i]", comm_order[i]);
            debug_int("transposed_dim.size_x[pidx_i]", transposed_dim.size_x[pidx_i]);
            debug_int("transposed_dim.size_z[pidx_j]", transposed_dim.size_z[pidx_j]);
            debug_int("output_dim.start_y[p_i]", output_dim.start_y[p_i]);
            debug_int("output_dim.size_y[p_i]", output_dim.size_y[p_i]);
            debug_int("comm_order[i]", comm_order[i]);
            debug_int("send_ptr offset", transposed_dim.size_x[pidx_i]*transposed_dim.size_z[pidx_j]*output_dim.start_y[p_i]);
            debug_int("size", sizeof(C_t)*transposed_dim.size_x[pidx_i]*output_dim.size_y[p_i]*transposed_dim.size_z[pidx_j]);
            debug_p("send_ptr", &send_ptr[transposed_dim.size_x[pidx_i]*transposed_dim.size_z[pidx_j]*output_dim.start_y[p_i]]);
            debug_h("\n");

            // After copy is complete, MPI starts a non-blocking send operation
            // Callback_Params<T, C_t> params = {i, send_ptr, this};
            // CUDA_CALL(cudaLaunchHostFunc(streams[p_i], this->MPIsend_Callback_SecondTranspose, (void *)&params));

            MPI_Isend(&send_ptr[transposed_dim.size_x[pidx_i]*transposed_dim.size_z[pidx_j]*output_dim.start_y[p_i]], 
                sizeof(C_t)*transposed_dim.size_x[pidx_i]*output_dim.size_y[p_i]*transposed_dim.size_z[pidx_j], MPI_BYTE,
                comm_order[i], pidx_i, comm, &(send_req[p_i]));
        }

        MPI_Waitall(partition->P1, send_req.data(), MPI_STATUSES_IGNORE);

        C_t *temp1_ptr = cuda_aware ? recv_ptr : temp_ptr;
        // Transpose local block and copy it to the recv buffer
        // Here, we use the recv buffer instead of the temp buffer, as the received data is already correctly aligned
        {
            debug_h("\nSecond Local Transpose\n");
            debug_int("transposed_dim.size_x[pidx_i]", transposed_dim.size_x[pidx_i]);
            debug_int("transposed_dim.size_y[0]", transposed_dim.size_y[0]);
            debug_int("transposed_dim.size_z[pidx_j]", transposed_dim.size_z[pidx_j]);
            debug_int("transposed_dim.start_x[pidx_i]", transposed_dim.start_x[pidx_i]);
            debug_int("output_dim.size_y[pidx_i]", output_dim.size_y[pidx_i]);
            debug_int("output_dim.size_z[pidx_j]", output_dim.size_z[pidx_j]);
            debug_int("output_dim.start_y[pidx_i]", output_dim.start_y[pidx_i]);
            debug_h("\n");

            cudaMemcpy3DParms cpy_params = {0};
            cpy_params.srcPos = make_cudaPos(0, output_dim.start_y[pidx_i], 0);
            cpy_params.srcPtr = make_cudaPitchedPtr(complex, sizeof(C_t)*transposed_dim.size_z[pidx_j], transposed_dim.size_z[pidx_j], transposed_dim.size_y[0]);
            cpy_params.dstPos = make_cudaPos(0, 0, transposed_dim.start_x[pidx_i]);
            cpy_params.dstPtr = make_cudaPitchedPtr(temp1_ptr, sizeof(C_t)*output_dim.size_z[pidx_j], output_dim.size_z[pidx_j], output_dim.size_y[pidx_i]);
            cpy_params.extent = make_cudaExtent(sizeof(C_t)*transposed_dim.size_z[pidx_j], output_dim.size_y[pidx_i], transposed_dim.size_x[pidx_i]);
            cpy_params.kind   = cudaMemcpyDeviceToDevice;

            CUDA_CALL(cudaMemcpy3DAsync(&cpy_params, streams[pidx_i]));            
        }

        // Start copying the received blocks to GPU memory (if !cuda_aware)
        // Otherwise the data is already correctly aligned. Therefore, we compute the last 1D FFT (x-direction) in the recv buffer (= temp1 buffer)
        if (!cuda_aware){
            do {
                // recv_req contains one null handle (i.e. recv_req[pidx_i]) and P1-1 active handles
                // If all active handles are processed, Waitany will return MPI_UNDEFINED
                MPI_Waitany(partition->P1, recv_req.data(), &p, MPI_STATUSES_IGNORE);
                if (p == MPI_UNDEFINED)
                    break;

                debug_h("\nSecond Global Receive\n");
                debug_int("p", p);
                debug_int("transposed_dim.size_x[p]", transposed_dim.size_x[p]);
                debug_int("transposed_dim.start_x[p]", transposed_dim.start_x[p]);
                debug_int("output_dim.size_z[pidx_j]", output_dim.size_z[pidx_j]);
                debug_int("output_dim.size_y[pidx_i]", output_dim.size_y[pidx_i]);
                debug_int("temp1_ptr offset", transposed_dim.start_x[p]*output_dim.size_y[pidx_i]*output_dim.size_z[pidx_j]);
                debug_int("recv_ptr offset", transposed_dim.start_x[p]*output_dim.size_y[pidx_i]*output_dim.size_z[pidx_j]);
                debug_int("size", sizeof(C_t)*transposed_dim.size_x[p]*output_dim.size_y[pidx_i]*output_dim.size_z[pidx_j]);
                
                // At this point, we received data of one of the P1-1 other relevant processes
                CUDA_CALL(cudaMemcpyAsync(&temp1_ptr[transposed_dim.start_x[p]*output_dim.size_y[pidx_i]*output_dim.size_z[pidx_j]],
                    &recv_ptr[transposed_dim.start_x[p]*output_dim.size_y[pidx_i]*output_dim.size_z[pidx_j]], 
                    sizeof(C_t)*transposed_dim.size_x[p]*output_dim.size_y[pidx_i]*output_dim.size_z[pidx_j], cudaMemcpyHostToDevice, 
                    streams[(pidx_i + partition->P1 - p) % partition->P1])); // TODO: check if this is the best stream for selection!
            } while (p != MPI_UNDEFINED);
        } else {
            MPI_Waitall(partition->P1, recv_req.data(), MPI_STATUSES_IGNORE);
        }
        // Wait for memcpy to complete
        CUDA_CALL(cudaDeviceSynchronize());

        // Compute the 1D FFT in x-direction
        CUFFT_CALL(cuFFT<T>::execC2C(planC2C_1, temp1_ptr, complex, CUFFT_FORWARD));
        CUDA_CALL(cudaDeviceSynchronize());
    }
}

template class MPIcuFFT_Pencil<float>;
template class MPIcuFFT_Pencil<double>;