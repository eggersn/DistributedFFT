#include "mpicufft_slab.hpp"
#include "cufft.hpp"
#include <cuda_runtime_api.h>


#define CUDA_CALL(x) do { if((x)!=cudaSuccess) {        \
    printf("Error %d at %s:%d\n",x,__FILE__,__LINE__);  \
    exit(EXIT_FAILURE); }} while(0)

#define CUFFT_CALL(x) do { if((x)!=CUFFT_SUCCESS) {     \
    printf("Error %d at %s:%d\n",x,__FILE__,__LINE__);  \
    exit(EXIT_FAILURE); }} while(0)

#define DEBUG 1
#define debug(d, v) {                                                 \
  if (DEBUG == 1) {                                                   \
    printf("DEBUG (%d): %s: %s in %s:%d\n",pidx,d,v,__FILE__,__LINE__); \
  }                                                                   \
}

#define debug_h(v) {                                                  \
  if (DEBUG == 1) {                                                   \
    printf("%s",v);                \
  }                                                                   \
}

#define debug_int(d, v) {                                             \
  if (DEBUG == 1) {                                                   \
    printf("DEBUG (%d): %s: %d in %s:%d\n",pidx,d,v,__FILE__,__LINE__); \
  }                                                                   \
}

#define debug_p(d, v) {                                                  \
  if (DEBUG == 1) {                                                   \
    printf("DEBUG (%d): %s: %p in %s:%d\n",pidx,d,v,__FILE__,__LINE__); \
  }                                                                   \
}

template<typename T> 
MPIcuFFT_Slab<T>::MPIcuFFT_Slab(MPI_Comm comm, bool mpi_cuda_aware, int max_world_size) : MPIcuFFT<T>(comm, mpi_cuda_aware, max_world_size) {
  isizex.resize(pcnt, 0);
  istartx.resize(pcnt, 0);
  osizey.resize(pcnt, 0);
  ostarty.resize(pcnt, 0);

  send_req.resize(pcnt, MPI_REQUEST_NULL);
  recv_req.resize(pcnt, MPI_REQUEST_NULL);

  isizez = 0;
  osizez = 0;

  planR2C = 0;
  planC2C = 0;

  if (pcnt%2 == 1) {
    for (int i=0; i<pcnt; ++i){
        if ((pcnt+i-pidx)%pcnt != pidx)
            comm_order.push_back((pcnt+i-pidx)%pcnt);
    }
  } else if (((pcnt-1)&pcnt) == 0) {
      for (int i=1; i<pcnt; ++i)
          comm_order.push_back(pidx^i);
  } else {
      for (int i=0; i<pcnt-1;++i) {
          int idle = (pcnt*i/2)%(pcnt-1);
          if (pidx == pcnt-1) 
              comm_order.push_back(idle);
          else if (pidx == idle) 
              comm_order.push_back(pcnt-1);
          else 
              comm_order.push_back((pcnt+i-pidx-1) % (pcnt-1));
      }
  }

  for (int i = 0; i < pcnt; i++){
    cudaStream_t stream;
    CUDA_CALL(cudaStreamCreate(&stream));
    streams.push_back(stream);
  }
}

template<typename T> 
MPIcuFFT_Slab<T>::~MPIcuFFT_Slab() {
    if (planR2C) 
        CUFFT_CALL(cufftDestroy(planR2C));
    if (planC2C) 
        CUFFT_CALL(cufftDestroy(planC2C));
}

template<typename T>
void MPIcuFFT_Slab<T>::initFFT(GlobalSize *global_size, Partition *partition, bool allocate) {
  using R_t = typename cuFFT<T>::R_t;
  using C_t = typename cuFFT<T>::C_t;

  // isizex stores how the input 3d array is distributed among the mpi processes
  size_t N1    = global_size->Nx / pcnt;
  size_t N1mod = global_size->Nx % pcnt;
  for (int p = 0; p < pcnt; ++p) {
    isizex[p]  = N1 + ((static_cast<size_t>(p) < N1mod) ? 1 : 0);
    istartx[p] = ((p==0) ? 0 : istartx[p-1]+isizex[p-1]);
  }

  //we only divide across the x-axis
  isizey = global_size->Ny; isizez = global_size->Nz;
  
  //after transposing the array, it is divided across the y-axis
  size_t N2    = global_size->Ny / pcnt;
  size_t N2mod = global_size->Ny % pcnt;
  for (int p = 0; p < pcnt; ++p) {
    osizey[p]  = N2 + ((static_cast<size_t>(p) < N2mod) ? 1 : 0);
    ostarty[p] = ((p==0) ? 0 : ostarty[p-1]+osizey[p-1]);
  }
  //for real input values, the second half (of the z-axis) is symmetric to the first half
  osizex = global_size->Nx; osizez = (global_size->Nz / 2) + 1;
      
  domainsize = sizeof(C_t) * std::max(isizex[pidx]*isizey*((isizez/2) + 1), osizex*osizey[pidx]*osizez);
  
  //sizes of the different workspaces
  size_t ws_r2c, ws_c2r, ws_c2c;
  
  CUFFT_CALL(cufftCreate(&planR2C));
  CUFFT_CALL(cufftSetAutoAllocation(planR2C, 0));
  
  if (fft3d) { // combined 3d fft, in case only one mpi process is used
    CUFFT_CALL(cufftMakePlan3d(planR2C, global_size->Nx, global_size->Ny, global_size->Nz, cuFFT<T>::R2Ctype, &ws_r2c));

    fft_worksize = std::max(ws_r2c, ws_c2r);
  } else { // 2d slab decomposition fft
    size_t batch = isizex[pidx];
    
    //here, an additional C2C transform is needed
    CUFFT_CALL(cufftCreate(&planC2C));
    CUFFT_CALL(cufftSetAutoAllocation(planC2C, 0));
    
    // For the forward FFT, we start with with a 2D transform in y,z direction. Afterwards, we compute a 1D transform for the x-axis.
    long long n[3] = {static_cast<long long>(osizex), static_cast<long long>(isizey), static_cast<long long>(isizez)};
    long long nembed[1] = {1};
    
    // For the forward FFT, where we can use the default data layout (thus the NULL pointer, see cuFFT doc for more details)
    // Execution order: (1) -> (3)
    CUFFT_CALL(cufftMakePlanMany64(planR2C, 2, &n[1], 0, 0, 0, 0, 0, 0, cuFFT<T>::R2Ctype, batch, &ws_r2c));
    // Here, the offset of two subsequent elements (x-axis) have an offset of osizey[pidx]*osizez.
    // Assumption: Data Layout [x][y][z]
    CUFFT_CALL(cufftMakePlanMany64(planC2C, 1, n, nembed, osizey[pidx]*osizez, 1, nembed, osizey[pidx]*osizez, 1, cuFFT<T>::C2Ctype, osizey[pidx]*osizez, &ws_c2c));
    
    fft_worksize = std::max(ws_r2c, ws_c2r);
    fft_worksize = std::max(fft_worksize, ws_c2c);
  }
  
  if (fft_worksize < domainsize) 
    fft_worksize = domainsize;

  // worksize_d is split into 3 parts:
  // 1. space for received data, 2. space for transmitted data, 3. actual workspace (see "mem_d")
  worksize_d = fft_worksize + (fft3d ? 0 : 2*domainsize);
  // analogously for the host worksize, if mpi is not cuda-aware
  worksize_h = (cuda_aware || fft3d ? 0 : 2*domainsize);

  if (allocate) 
    this->setWorkArea();
  
  CUDA_CALL(cudaDeviceSynchronize());
}

//default parameters device=nullptr, host=nullptr
template<typename T> 
void MPIcuFFT_Slab<T>::setWorkArea(void *device, void *host) {
  if (!domainsize) 
    return;

  if (device && allocated_d) {
    CUDA_CALL(cudaFree(workarea_d));
    allocated_d = false;
    workarea_d = device;
  } else if (!allocated_d && device) {
    workarea_d = device;
  } else if (!allocated_d && !device) {
    CUDA_CALL(cudaMalloc(&(workarea_d), worksize_d));
    allocated_d = true;
  }

  mem_d.clear();
  for (size_t i=0; i< 1 + (fft3d ? 0 : (cuda_aware ? 2 : 1)); ++i) 
    mem_d.push_back(&static_cast<char*>(workarea_d)[i*domainsize]);
  
  if (fft3d) {
    CUFFT_CALL(cufftSetWorkArea(planR2C, mem_d[0]));
  } else {
    CUFFT_CALL(cufftSetWorkArea(planR2C, mem_d[cuda_aware ? 2 : 1]));
    CUFFT_CALL(cufftSetWorkArea(planC2C, mem_d[cuda_aware ? 2 : 1]));
  }
    
  if (host && allocated_h) {
    CUDA_CALL(cudaFreeHost(workarea_h));
    allocated_h = false;
    workarea_h = host;
  } else if (!allocated_h && host) {
    workarea_h = host;
  } else if (!allocated_h && !host && worksize_h) {
    CUDA_CALL(cudaMallocHost(&(workarea_h), worksize_h));
    allocated_h = true;
  }

  mem_h.clear();
  for (size_t i=0; i<((fft3d || cuda_aware) ? 0 : 2); ++i)
    mem_h.push_back(&static_cast<char*>(workarea_h)[i*domainsize]);

  initialized = true;
}

template<typename T> 
void MPIcuFFT_Slab<T>::MPIsend_Callback(void *data) {
  struct Callback_Params *params = (Callback_Params *)data;
  struct Callback_Params_Base *base_params = params->base_params;
  {
    std::lock_guard<std::mutex> lk(base_params->mutex);
    base_params->comm_ready.push_back(params->p);
  }
  base_params->cv.notify_one();
}

template<typename T>
void MPIcuFFT_Slab<T>::MPIsend_Thread(Callback_Params_Base &base_params, void *ptr) {
  using C_t = typename cuFFT<T>::C_t;
  C_t *send_ptr = (C_t *) ptr;

  for (int i = 0; i < comm_order.size(); i++){
    std::unique_lock<std::mutex> lk(base_params.mutex);
    base_params.cv.wait(lk, [&base_params]{return !base_params.comm_ready.empty();});

    int p = base_params.comm_ready.back();
    base_params.comm_ready.pop_back();
    size_t oslice = isizex[pidx]*osizez*ostarty[p];

    MPI_Isend(&send_ptr[oslice], 
              sizeof(C_t)*isizex[pidx]*osizez*osizey[p], MPI_BYTE, 
              p, pidx, comm, &(send_req[p]));
    lk.unlock();
  }
}

template<typename T> 
void MPIcuFFT_Slab<T>::execR2C(void *out, const void *in) {
  if (!initialized) 
    return;

  using R_t = typename cuFFT<T>::R_t;
  using C_t = typename cuFFT<T>::C_t;
  R_t *real    = cuFFT<T>::real(in);
  C_t *complex = cuFFT<T>::complex(out);
  if (fft3d) {
    CUFFT_CALL(cuFFT<T>::execR2C(planR2C, real, complex));
    CUDA_CALL(cudaDeviceSynchronize());
  } else {
    C_t *recv_ptr, *send_ptr, *temp_ptr;
    temp_ptr = cuFFT<T>::complex(mem_d[0]);
    if (cuda_aware) {
      recv_ptr = cuFFT<T>::complex(mem_d[0]); // = temp_ptr!
      send_ptr = cuFFT<T>::complex(mem_d[1]);
    } else {
      recv_ptr = cuFFT<T>::complex(mem_h[0]);
      send_ptr = cuFFT<T>::complex(mem_h[1]);
    }
    recv_req[pidx] = MPI_REQUEST_NULL;
    send_req[pidx] = MPI_REQUEST_NULL;

    // compute 2d FFT 
    CUFFT_CALL(cuFFT<T>::execR2C(planR2C, real, complex));
    CUDA_CALL(cudaDeviceSynchronize());

    /* We are interested in sending the block via MPI as soon as cudaMemcpy2DAsync is done.
    *  Therefore, MPIsend_Callback simulates a producer and MPIsend_Thread a consumer of a 
    *  channel with blocking receive (via conditional variable)
    */
    Callback_Params_Base base_params;
    std::vector<Callback_Params> params_array;

    for (int i = 0; i < pcnt; i++){
      Callback_Params params = {&base_params, i};
      params_array.push_back(params);
    }
  
    for (auto p : comm_order) { 
      // start non-blocking receive for rank p
      MPI_Irecv((&recv_ptr[istartx[p]*osizez*osizey[pidx]]),
        sizeof(C_t)*isizex[p]*osizez*osizey[pidx], MPI_BYTE,
        p, p, comm, &(recv_req[p]));

      size_t oslice = isizex[pidx]*osizez*ostarty[p];
 
      CUDA_CALL(cudaMemcpy2DAsync(&send_ptr[oslice], sizeof(C_t)*osizey[p]*osizez,
                                  &complex[ostarty[p]*osizez], sizeof(C_t)*isizey*osizez,
                                  sizeof(C_t)*osizey[p]*osizez, isizex[pidx],
                                  cuda_aware?cudaMemcpyDeviceToDevice:cudaMemcpyDeviceToHost, streams[p]));

      // Callback function for the specific stream
      CUDA_CALL(cudaLaunchHostFunc(streams[p], this->MPIsend_Callback, (void *)&params_array[p]));
    }
    // Thread which is used to send the MPI messages
    std::thread mpisend_thread(&MPIcuFFT_Slab<T>::MPIsend_Thread, this, std::ref(base_params), send_ptr);
    { 
      // transpose local block
      size_t oslice = osizez*osizey[pidx]*istartx[pidx];

      CUDA_CALL(cudaMemcpy2DAsync(&temp_ptr[oslice], sizeof(C_t)*osizey[pidx]*osizez,
                                  &complex[ostarty[pidx]*osizez], sizeof(C_t)*isizey*osizez, 
                                  sizeof(C_t)*osizey[pidx]*osizez, isizex[pidx],
                                  cudaMemcpyDeviceToDevice, streams[pidx]));
    }
    if (!cuda_aware) { // copy received blocks to device
      int p, i = 0;
      do {
        MPI_Waitany(pcnt, recv_req.data(), &p, MPI_STATUSES_IGNORE);
        if (p == MPI_UNDEFINED) break;
        CUDA_CALL(cudaMemcpyAsync(&temp_ptr[istartx[p]*osizez*osizey[pidx]],
                                  &recv_ptr[istartx[p]*osizez*osizey[pidx]],
                                  isizex[p]*osizez*osizey[pidx]*sizeof(C_t), cudaMemcpyHostToDevice, streams[comm_order[i]]));
        i++;
      } while(p != MPI_UNDEFINED);
    } else { // just wait for all receives
      MPI_Waitall(pcnt, recv_req.data(), MPI_STATUSES_IGNORE);
    }
    CUDA_CALL(cudaDeviceSynchronize());
    // compute remaining 1d FFT, for cuda-aware recv and temp buffer are identical
    CUFFT_CALL(cuFFT<T>::execC2C(planC2C, temp_ptr, complex, CUFFT_FORWARD));
    CUDA_CALL(cudaDeviceSynchronize());
    MPI_Waitall(pcnt, send_req.data(), MPI_STATUSES_IGNORE);
    mpisend_thread.join();
  }
}

template class MPIcuFFT_Slab<float>;
template class MPIcuFFT_Slab<double>;