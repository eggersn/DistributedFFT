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
MPIcuFFT_Slab<T>::MPIcuFFT_Slab(Configurations config, MPI_Comm comm, int max_world_size) : MPIcuFFT<T>(config, comm, max_world_size) {
  input_sizes_x.resize(pcnt, 0);
  input_start_x.resize(pcnt, 0);
  output_sizes_y.resize(pcnt, 0);
  output_start_y.resize(pcnt, 0);

  if (config.comm_method == Peer2Peer) {
    send_req.resize(pcnt, MPI_REQUEST_NULL);
    recv_req.resize(pcnt, MPI_REQUEST_NULL);
  }

  input_size_z = 0;
  output_size_z = 0;

  planR2C = 0;
  planC2C = 0;
  planC2R = 0;

  for (int i = 1; i < pcnt; i++)
    comm_order.push_back((pidx + i) % pcnt);

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
    if (planC2R) 
        CUFFT_CALL(cufftDestroy(planC2R));

    delete timer;
}

template<typename T>
void MPIcuFFT_Slab<T>::initFFT(GlobalSize *global_size, Partition *partition, bool allocate) {
  mkdir((config.benchmark_dir +  "/slab_default").c_str(), 0777);
  std::stringstream ss;
  ss << config.benchmark_dir <<  "/slab_default/test_0_" << config.comm_method << "_" << config.send_method << "_" << global_size->Nx << "_" << global_size->Ny << "_" << global_size->Nz;
  ss << "_" << cuda_aware << "_" << pcnt << ".csv";
  std::string filename = ss.str();

  timer = new Timer(comm, 0, pcnt, pidx, section_descriptions, filename);

  timer->start();
  using R_t = typename cuFFT<T>::R_t;
  using C_t = typename cuFFT<T>::C_t;

  // input_sizes_x stores how the input 3d array is distributed among the mpi processes
  size_t N1    = global_size->Nx / pcnt;
  size_t N1mod = global_size->Nx % pcnt;
  for (int p = 0; p < pcnt; ++p) {
    input_sizes_x[p]  = N1 + ((static_cast<size_t>(p) < N1mod) ? 1 : 0);
    input_start_x[p] = ((p==0) ? 0 : input_start_x[p-1]+input_sizes_x[p-1]);
  }

  //we only divide across the x-axis
  input_size_y = global_size->Ny; input_size_z = global_size->Nz;
  
  //after transposing the array, it is divided across the y-axis
  size_t N2    = global_size->Ny / pcnt;
  size_t N2mod = global_size->Ny % pcnt;
  for (int p = 0; p < pcnt; ++p) {
    output_sizes_y[p]  = N2 + ((static_cast<size_t>(p) < N2mod) ? 1 : 0);
    output_start_y[p] = ((p==0) ? 0 : output_start_y[p-1]+output_sizes_y[p-1]);
  }
  //for real input values, the second half (of the z-axis) is symmetric to the first half
  output_size_x = global_size->Nx; output_size_z = (global_size->Nz / 2) + 1;
      
  domainsize = sizeof(C_t) * std::max(input_sizes_x[pidx]*input_size_y*output_size_z, output_size_x*output_sizes_y[pidx]*output_size_z);
  
  //sizes of the different workspaces
  size_t ws_r2c, ws_c2c, ws_c2r;
  
  CUFFT_CALL(cufftCreate(&planR2C));
  CUFFT_CALL(cufftSetAutoAllocation(planR2C, 0));
  CUFFT_CALL(cufftCreate(&planC2R));
  CUFFT_CALL(cufftSetAutoAllocation(planC2R, 0));
  
  if (fft3d) { // combined 3d fft, in case only one mpi process is used
    CUFFT_CALL(cufftMakePlan3d(planR2C, global_size->Nx, global_size->Ny, global_size->Nz, cuFFT<T>::R2Ctype, &ws_r2c));
    CUFFT_CALL(cufftMakePlan3d(planC2R, global_size->Nx, global_size->Ny, global_size->Nz, cuFFT<T>::C2Rtype, &ws_c2r));
    fft_worksize = std::max(ws_r2c, ws_c2r);
  } else { // 2d slab decomposition fft
    size_t batch = input_sizes_x[pidx];
    
    //here, an additional C2C transform is needed
    CUFFT_CALL(cufftCreate(&planC2C));
    CUFFT_CALL(cufftSetAutoAllocation(planC2C, 0));
    
    // For the forward FFT, we start with with a 2D transform in y,z direction. Afterwards, we compute a 1D transform for the x-axis.
    long long n[3] = {static_cast<long long>(output_size_x), static_cast<long long>(input_size_y), static_cast<long long>(input_size_z)};
    long long nembed[1] = {1};
    
    // For the forward FFT, where we can use the default data layout (thus the NULL pointer, see cuFFT doc for more details)
    // Execution order: (1) -> (3)
    CUFFT_CALL(cufftMakePlanMany64(planR2C, 2, &n[1], 0, 0, 0, 0, 0, 0, cuFFT<T>::R2Ctype, batch, &ws_r2c));
    // Here, the offset of two subsequent elements (x-axis) have an offset of output_sizes_y[pidx]*output_size_z.
    // Assumption: Data Layout [x][y][z]
    CUFFT_CALL(cufftMakePlanMany64(planC2C, 1, n, nembed, output_sizes_y[pidx]*output_size_z, 1, 
      nembed, output_sizes_y[pidx]*output_size_z, 1, cuFFT<T>::C2Ctype, output_sizes_y[pidx]*output_size_z, &ws_c2c));

    CUFFT_CALL(cufftMakePlanMany64(planC2R, 2, &n[1], 0, 0, 0, 0, 0, 0, cuFFT<T>::C2Rtype, batch, &ws_c2r));
    
    fft_worksize = std::max(std::max(ws_r2c, ws_c2c), ws_c2r);
  }
  
  if (fft_worksize < domainsize) 
    fft_worksize = domainsize;

  // worksize_d is split into 3 parts:
  // 1. space for received data, 2. space for transmitted data, 3. actual workspace (see "mem_d")
  worksize_d = fft_worksize + (fft3d ? 0 : (config.send_method == MPI_Type || !cuda_aware ? domainsize : 2*domainsize));

  // analogously for the host worksize, if mpi is not cuda-aware
  worksize_h = (cuda_aware || fft3d ? 0 : 2*domainsize);

  if (allocate) 
    this->setWorkArea();

  if (config.comm_method == Peer2Peer) {
    if (config.send_method == Streams) {
      /* We are interested in sending the block via MPI as soon as cudaMemcpy2DAsync is done.
      *  Therefore, MPIsend_Callback simulates a producer and MPIsend_Thread a consumer of a 
      *  channel with blocking receive (via conditional variable)
      */
      for (int i = 0; i < pcnt; i++){
        Callback_Params params = {&base_params, i};
        params_array.push_back(params);
      }
    } else if (config.send_method == MPI_Type) {
      MPI_PENCILS = std::vector<MPI_Datatype>(pcnt);
      for (int i = 0; i < pcnt; i++) {
          MPI_Type_vector(input_sizes_x[pidx], output_size_z*output_sizes_y[i]*sizeof(C_t), output_size_z*input_size_y*sizeof(C_t), MPI_BYTE, &MPI_PENCILS[i]);
          MPI_Type_commit(&MPI_PENCILS[i]);
      }
    }
  } else if (config.comm_method == All2All) {
    if (config.send_method == MPI_Type) {
      MPI_PENCILS = std::vector<MPI_Datatype>(pcnt);
      MPI_RECV = std::vector<MPI_Datatype>(pcnt);

      sendcounts = std::vector<int>(pcnt, 1);
      sdispls = std::vector<int>(pcnt, 0);
      recvcounts = std::vector<int>(pcnt, 0);
      rdispls = std::vector<int>(pcnt, 0);
      for (int p = 0; p < pcnt; p++) {
        sdispls[p] = output_size_z*output_start_y[p]*sizeof(C_t);
        recvcounts[p] = output_size_z*output_sizes_y[pidx]*input_sizes_x[p]*sizeof(C_t);
        rdispls[p] = output_size_z*output_sizes_y[pidx]*input_start_x[p]*sizeof(C_t);
        MPI_Type_vector(input_sizes_x[pidx], output_size_z*output_sizes_y[p]*sizeof(C_t), output_size_z*input_size_y*sizeof(C_t), MPI_BYTE, &MPI_PENCILS[p]);
        MPI_Type_commit(&MPI_PENCILS[p]);
        MPI_RECV[p] = MPI_BYTE;
      }
    } else {
      sendcounts = std::vector<int>(pcnt, 0);
      sdispls = std::vector<int>(pcnt, 0);
      recvcounts = std::vector<int>(pcnt, 0);
      rdispls = std::vector<int>(pcnt, 0);
      for (int p = 0; p < pcnt; p++) {
        sendcounts[p] = output_size_z*output_sizes_y[p]*input_sizes_x[pidx]*sizeof(C_t);
        sdispls[p] = output_size_z*input_sizes_x[pidx]*output_start_y[p]*sizeof(C_t);
        recvcounts[p] = output_size_z*output_sizes_y[pidx]*input_sizes_x[p]*sizeof(C_t);
        rdispls[p] = output_size_z*output_sizes_y[pidx]*input_start_x[p]*sizeof(C_t);
      }
    }
  }
  
  CUDA_CALL(cudaDeviceSynchronize());
  timer->stop_store("init");
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
  for (size_t i=0; i< 1 + (fft3d ? 0 : (!cuda_aware || config.send_method == MPI_Type ? 1 : 2)); ++i) 
    mem_d.push_back(&static_cast<char*>(workarea_d)[i*domainsize]);
  
  if (fft3d) {
    CUFFT_CALL(cufftSetWorkArea(planR2C, mem_d[0]));
    CUFFT_CALL(cufftSetWorkArea(planC2R, mem_d[0]));
  } else {
    CUFFT_CALL(cufftSetWorkArea(planR2C, mem_d[!cuda_aware || config.send_method == MPI_Type ? 1 : 2]));
    CUFFT_CALL(cufftSetWorkArea(planC2R, mem_d[!cuda_aware || config.send_method == MPI_Type ? 1 : 2]));
    CUFFT_CALL(cufftSetWorkArea(planC2C, mem_d[!cuda_aware || config.send_method == MPI_Type ? 1 : 2]));
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
void MPIcuFFT_Slab<T>::Peer2Peer_Sync(void *complex_, void *recv_ptr_, bool forward) {
  using C_t = typename cuFFT<T>::C_t;
  C_t *complex = cuFFT<T>::complex(complex_);
  C_t *recv_ptr = cuFFT<T>::complex(recv_ptr_);
  C_t *send_ptr;

  if (forward) {
    if (cuda_aware)
      send_ptr = cuFFT<T>::complex(mem_d[1]);
    else 
      send_ptr = cuFFT<T>::complex(mem_h[1]);

    for (auto p : comm_order) { 
      // start non-blocking receive for rank p
      MPI_Irecv((&recv_ptr[input_start_x[p]*output_size_z*output_sizes_y[pidx]]),
                  sizeof(C_t)*input_sizes_x[p]*output_size_z*output_sizes_y[pidx], MPI_BYTE,
                  p, p, comm, &(recv_req[p]));

      size_t oslice = input_sizes_x[pidx]*output_size_z*output_start_y[p];

      CUDA_CALL(cudaMemcpy2DAsync(&send_ptr[oslice], sizeof(C_t)*output_sizes_y[p]*output_size_z,
                                  &complex[output_start_y[p]*output_size_z], sizeof(C_t)*input_size_y*output_size_z,
                                  sizeof(C_t)*output_sizes_y[p]*output_size_z, input_sizes_x[pidx],
                                  cuda_aware?cudaMemcpyDeviceToDevice:cudaMemcpyDeviceToHost, streams[p]));
      CUDA_CALL(cudaDeviceSynchronize());
      if (p == comm_order[0])
        timer->stop_store("Transpose (First Send)");

      MPI_Isend(&send_ptr[oslice], 
                sizeof(C_t)*input_sizes_x[pidx]*output_size_z*output_sizes_y[p], MPI_BYTE, 
                p, pidx, comm, &(send_req[p]));
    }
    timer->stop_store("Transpose (Packing)");
  } else {
    C_t* temp_ptr = cuFFT<T>::complex(mem_d[0]);
    if (cuda_aware)
      send_ptr = cuFFT<T>::complex(mem_d[0]);
    else 
      send_ptr = cuFFT<T>::complex(mem_h[1]);

    if (!cuda_aware) {
      CUDA_CALL(cudaMemcpyAsync(send_ptr, temp_ptr, output_size_x*output_sizes_y[pidx]*output_size_z*sizeof(C_t), cudaMemcpyDeviceToHost));
      CUDA_CALL(cudaDeviceSynchronize());
    }
    timer->stop_store("Transpose (Packing)");
    for (auto p : comm_order) {
      MPI_Irecv(&recv_ptr[input_sizes_x[pidx]*output_size_z*output_start_y[p]], 
        sizeof(C_t)*input_sizes_x[pidx]*output_size_z*output_sizes_y[p], MPI_BYTE, p, p, comm, &recv_req[p]);

      if (p == comm_order[0])
        timer->stop_store("Transpose (First Send)");  

      MPI_Isend(&send_ptr[input_start_x[p]*output_sizes_y[pidx]*output_size_z], 
        input_sizes_x[p]*output_sizes_y[pidx]*output_size_z*sizeof(C_t), MPI_BYTE, p, pidx, comm, &send_req[p]);
    }
  }
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

    if (i == 0)
      timer->stop_store("Transpose (First Send)");

    if (forward) {
      size_t oslice = input_sizes_x[pidx]*output_size_z*output_start_y[p];
      MPI_Isend(&send_ptr[oslice], 
        sizeof(C_t)*input_sizes_x[pidx]*output_size_z*output_sizes_y[p], MPI_BYTE, 
        p, pidx, comm, &(send_req[p]));
    } else {
      MPI_Isend(&send_ptr[input_start_x[p]*output_sizes_y[pidx]*output_size_z], 
        input_sizes_x[p]*output_sizes_y[pidx]*output_size_z*sizeof(C_t), MPI_BYTE, 
        p, pidx, comm, &send_req[p]);
    }

    lk.unlock();
  }
  timer->stop_store("Transpose (Packing)");
}

template<typename T>
void MPIcuFFT_Slab<T>::Peer2Peer_Streams(void *complex_, void *recv_ptr_, bool forward) {
  using C_t = typename cuFFT<T>::C_t;
  C_t *complex = cuFFT<T>::complex(complex_);
  C_t *recv_ptr = cuFFT<T>::complex(recv_ptr_);
  C_t *send_ptr;

  if (forward) {
    if (cuda_aware)
      send_ptr = cuFFT<T>::complex(mem_d[1]);
    else 
      send_ptr = cuFFT<T>::complex(mem_h[1]);

    // Thread which is used to send the MPI messages
    mpisend_thread = std::thread(&MPIcuFFT_Slab<T>::MPIsend_Thread, this, std::ref(base_params), send_ptr);

    for (auto p : comm_order) { 
      // start non-blocking receive for rank p
      MPI_Irecv((&recv_ptr[input_start_x[p]*output_size_z*output_sizes_y[pidx]]),
      sizeof(C_t)*input_sizes_x[p]*output_size_z*output_sizes_y[pidx], MPI_BYTE,
      p, p, comm, &(recv_req[p]));

      size_t oslice = input_sizes_x[pidx]*output_size_z*output_start_y[p];

      CUDA_CALL(cudaMemcpy2DAsync(&send_ptr[oslice], sizeof(C_t)*output_sizes_y[p]*output_size_z,
                                  &complex[output_start_y[p]*output_size_z], sizeof(C_t)*input_size_y*output_size_z,
                                  sizeof(C_t)*output_sizes_y[p]*output_size_z, input_sizes_x[pidx],
                                  cuda_aware?cudaMemcpyDeviceToDevice:cudaMemcpyDeviceToHost, streams[p]));

      // Callback function for the specific stream
      CUDA_CALL(cudaLaunchHostFunc(streams[p], this->MPIsend_Callback, (void *)&params_array[p]));
    }
  } else {
    C_t* temp_ptr = cuFFT<T>::complex(mem_d[0]);
    if (cuda_aware)
      send_ptr = temp_ptr;
    else 
      send_ptr = cuFFT<T>::complex(mem_h[1]);

    // Thread which is used to send the MPI messages
    if (!cuda_aware)
      mpisend_thread = std::thread(&MPIcuFFT_Slab<T>::MPIsend_Thread, this, std::ref(base_params), send_ptr);

    for (auto p : comm_order) {
      MPI_Irecv(&recv_ptr[input_sizes_x[pidx]*output_size_z*output_start_y[p]], 
        sizeof(C_t)*input_sizes_x[pidx]*output_size_z*output_sizes_y[p], MPI_BYTE, p, p, comm, &recv_req[p]);

      if (!cuda_aware) {
        CUDA_CALL(cudaMemcpyAsync(&send_ptr[input_start_x[p]*output_sizes_y[pidx]*output_size_z], 
          &temp_ptr[input_start_x[p]*output_sizes_y[pidx]*output_size_z], input_sizes_x[p]*output_sizes_y[pidx]*output_size_z*sizeof(C_t), 
          cudaMemcpyDeviceToHost, streams[p]));

        CUDA_CALL(cudaLaunchHostFunc(streams[p], this->MPIsend_Callback, (void *)&params_array[p]));
      } else {
        if (p == comm_order[0])
          timer->stop_store("Transpose (First Send)");  

        MPI_Isend(&send_ptr[input_start_x[p]*output_sizes_y[pidx]*output_size_z], 
          input_sizes_x[p]*output_sizes_y[pidx]*output_size_z*sizeof(C_t), MPI_BYTE, p, pidx, comm, &send_req[p]);
      }
    }
  }
}

template<typename T>
void MPIcuFFT_Slab<T>::Peer2Peer_MPIType(void *complex_, void *recv_ptr_, bool forward) {
  using C_t = typename cuFFT<T>::C_t;
  C_t *complex = cuFFT<T>::complex(complex_);
  C_t *recv_ptr = cuFFT<T>::complex(recv_ptr_);
  C_t *send_ptr;

  if (forward) {
    if (cuda_aware)
      send_ptr = complex;
    else 
      send_ptr = cuFFT<T>::complex(mem_h[1]);

    if (!cuda_aware) {
      CUDA_CALL(cudaMemcpyAsync(send_ptr, complex, output_size_z*input_size_y*input_sizes_x[pidx]*sizeof(C_t), cudaMemcpyDeviceToHost));
      CUDA_CALL(cudaDeviceSynchronize());
    }
    timer->stop_store("Transpose (Packing)");

    for (auto p : comm_order) { 
      // start non-blocking receive for rank p
      MPI_Irecv((&recv_ptr[input_start_x[p]*output_size_z*output_sizes_y[pidx]]),
                  sizeof(C_t)*input_sizes_x[p]*output_size_z*output_sizes_y[pidx], MPI_BYTE,
                  p, p, comm, &(recv_req[p]));
      if (p == comm_order[0])
        timer->stop_store("Transpose (First Send)");

      MPI_Isend(&send_ptr[output_size_z*output_start_y[p]], 1, MPI_PENCILS[p], p, pidx, comm, &send_req[p]);    
    }
  } else {
    C_t* temp_ptr = cuFFT<T>::complex(mem_d[0]);
    if (cuda_aware)
      send_ptr = cuFFT<T>::complex(mem_d[0]); // = temp_ptr
    else 
      send_ptr = cuFFT<T>::complex(mem_h[1]);

    if (!cuda_aware) {
      CUDA_CALL(cudaMemcpyAsync(send_ptr, temp_ptr, output_size_x*output_sizes_y[pidx]*output_size_z*sizeof(C_t), cudaMemcpyDeviceToHost));
      CUDA_CALL(cudaDeviceSynchronize());
    }
    timer->stop_store("Transpose (Packing)");

    for (auto p : comm_order) {
      MPI_Irecv(&recv_ptr[output_size_z*output_start_y[p]], 1, MPI_PENCILS[p], p, p, comm, &recv_req[p]);

      if (p == comm_order[0])
        timer->stop_store("Transpose (First Send)");  

      MPI_Isend(&send_ptr[input_start_x[p]*output_sizes_y[pidx]*output_size_z], 
        input_sizes_x[p]*output_sizes_y[pidx]*output_size_z*sizeof(C_t), MPI_BYTE, p, pidx, comm, &send_req[p]);
    }
  }
}

template<typename T>
void MPIcuFFT_Slab<T>::Peer2Peer_Communication(void *complex_, bool forward) {
  using C_t = typename cuFFT<T>::C_t;
  C_t *complex = cuFFT<T>::complex(complex_);

  // Forward FFT
  if (forward) {
    C_t *recv_ptr, *temp_ptr;
    temp_ptr = cuFFT<T>::complex(mem_d[0]);
    if (cuda_aware)
      recv_ptr = temp_ptr;
    else 
      recv_ptr = cuFFT<T>::complex(mem_h[0]);

    if (config.send_method == Sync)
      this->Peer2Peer_Sync(complex_, (void *)recv_ptr);
    else if (config.send_method == Streams)
      this->Peer2Peer_Streams(complex_, (void *)recv_ptr);
    else
      this->Peer2Peer_MPIType(complex_, (void *)recv_ptr);

    timer->stop_store("Transpose (Start Local Transpose)");
    { 
      // transpose local block
      size_t oslice = output_size_z*output_sizes_y[pidx]*input_start_x[pidx];

      CUDA_CALL(cudaMemcpy2DAsync(&temp_ptr[oslice], sizeof(C_t)*output_sizes_y[pidx]*output_size_z,
                                  &complex[output_start_y[pidx]*output_size_z], sizeof(C_t)*input_size_y*output_size_z, 
                                  sizeof(C_t)*output_sizes_y[pidx]*output_size_z, input_sizes_x[pidx],
                                  cudaMemcpyDeviceToDevice, streams[pidx]));
    }
    timer->stop_store("Transpose (Start Receive)");
    if (!cuda_aware) { // copy received blocks to device
      int p;
      do {
        MPI_Waitany(pcnt, recv_req.data(), &p, MPI_STATUSES_IGNORE);
        if (p == MPI_UNDEFINED) break;
        CUDA_CALL(cudaMemcpyAsync(&temp_ptr[input_start_x[p]*output_size_z*output_sizes_y[pidx]],
                                  &recv_ptr[input_start_x[p]*output_size_z*output_sizes_y[pidx]],
                                  input_sizes_x[p]*output_size_z*output_sizes_y[pidx]*sizeof(C_t), cudaMemcpyHostToDevice, streams[p]));
      } while(p != MPI_UNDEFINED);
    } else { // just wait for all receives
      MPI_Waitall(pcnt, recv_req.data(), MPI_STATUSES_IGNORE);
    }
    CUDA_CALL(cudaDeviceSynchronize());
    timer->stop_store("Transpose (Finished Receive)");
  } else { // Inverse FFT
    C_t *recv_ptr, *copy_ptr;
    C_t *temp_ptr = cuFFT<T>::complex(mem_d[0]);
    if (!cuda_aware) {
      copy_ptr = complex;
      recv_ptr = cuFFT<T>::complex(mem_h[0]);
    }

    if (config.send_method == MPI_Type) {
      if (cuda_aware)
        recv_ptr = complex;

      this->Peer2Peer_MPIType(complex_, (void *)recv_ptr, false);

      //local transpose
      timer->stop_store("Transpose (Start Local Transpose)");
      CUDA_CALL(cudaMemcpy2DAsync(&recv_ptr[output_start_y[pidx]*output_size_z], sizeof(C_t)*input_size_y*output_size_z,
        &temp_ptr[input_start_x[pidx]*output_sizes_y[pidx]*output_size_z], sizeof(C_t)*output_sizes_y[pidx]*output_size_z,
        sizeof(C_t)*output_sizes_y[pidx]*output_size_z, input_sizes_x[pidx], cuda_aware?cudaMemcpyDeviceToDevice:cudaMemcpyDeviceToHost));

      timer->stop_store("Transpose (Start Receive)");

      MPI_Waitall(pcnt, recv_req.data(), MPI_STATUSES_IGNORE);
      CUDA_CALL(cudaDeviceSynchronize());

      if (!cuda_aware) {
        CUDA_CALL(cudaMemcpyAsync(copy_ptr, recv_ptr, sizeof(C_t)*input_sizes_x[pidx]*input_size_y*output_size_z, cudaMemcpyHostToDevice));
        CUDA_CALL(cudaDeviceSynchronize());
      }
      timer->stop_store("Transpose (Finished Receive)");
    } else {
      if (cuda_aware) {
        copy_ptr = complex;
        recv_ptr = cuFFT<T>::complex(mem_d[1]); 
      }

      if (config.send_method == Sync)
        this->Peer2Peer_Sync(complex_, (void *)recv_ptr, false);
      else if (config.send_method == Streams)
        this->Peer2Peer_Streams(complex_, (void *)recv_ptr, false);
    
      //local transpose
      timer->stop_store("Transpose (Start Local Transpose)");
      {
        CUDA_CALL(cudaMemcpy2DAsync(&copy_ptr[output_start_y[pidx]*output_size_z], sizeof(C_t)*input_size_y*output_size_z,
          &temp_ptr[input_start_x[pidx]*output_sizes_y[pidx]*output_size_z], sizeof(C_t)*output_sizes_y[pidx]*output_size_z,
          sizeof(C_t)*output_sizes_y[pidx]*output_size_z, input_sizes_x[pidx], cudaMemcpyDeviceToDevice, streams[pidx]));
      }

      timer->stop_store("Transpose (Start Receive)");
      int p;
      do {
        MPI_Waitany(pcnt, recv_req.data(), &p, MPI_STATUSES_IGNORE);
        if (p == MPI_UNDEFINED) 
          break;

        CUDA_CALL(cudaMemcpy2DAsync(&copy_ptr[output_start_y[p]*output_size_z], sizeof(C_t)*input_size_y*output_size_z,
          &recv_ptr[input_sizes_x[pidx]*output_start_y[p]*output_size_z], sizeof(C_t)*output_sizes_y[p]*output_size_z,
          sizeof(C_t)*output_sizes_y[p]*output_size_z, input_sizes_x[pidx], cuda_aware ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice, streams[p]));
      } while (p != MPI_UNDEFINED);
      CUDA_CALL(cudaDeviceSynchronize());
      timer->stop_store("Transpose (Finished Receive)");
    }
  }
}

template<typename T>
void MPIcuFFT_Slab<T>::All2All_Sync(void *complex_, bool forward) {
  using C_t = typename cuFFT<T>::C_t;
  C_t *complex = cuFFT<T>::complex(complex_);
  C_t *send_ptr, *recv_ptr, *temp_ptr;

  if (forward) {
    temp_ptr = cuFFT<T>::complex(mem_d[0]);
    if (cuda_aware) {
      recv_ptr = temp_ptr;
      send_ptr = cuFFT<T>::complex(mem_d[1]);
    } else {
      recv_ptr = cuFFT<T>::complex(mem_h[0]);
      send_ptr = cuFFT<T>::complex(mem_h[1]);
    }

    for (int p = 0; p < pcnt; p++) { 
        if (p == pidx)
          timer->stop_store("Transpose (Start Local Transpose)");

        size_t oslice = input_sizes_x[pidx]*output_size_z*output_start_y[p];

        CUDA_CALL(cudaMemcpy2DAsync(&send_ptr[oslice], sizeof(C_t)*output_sizes_y[p]*output_size_z,
                                    &complex[output_start_y[p]*output_size_z], sizeof(C_t)*input_size_y*output_size_z,
                                    sizeof(C_t)*output_sizes_y[p]*output_size_z, input_sizes_x[pidx],
                                    cuda_aware?cudaMemcpyDeviceToDevice:cudaMemcpyDeviceToHost, streams[p]));
    }
    
    CUDA_CALL(cudaDeviceSynchronize());
    timer->stop_store("Transpose (Packing)");

    timer->stop_store("Transpose (Start All2All)");
    MPI_Alltoallv(send_ptr, sendcounts.data(), sdispls.data(), MPI_BYTE, 
                  recv_ptr, recvcounts.data(), rdispls.data(), MPI_BYTE, comm);
    timer->stop_store("Transpose (Finished All2All)");

    if (!cuda_aware) {
      CUDA_CALL(cudaMemcpyAsync(temp_ptr, recv_ptr, output_size_x*output_sizes_y[pidx]*output_size_z*sizeof(C_t), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaDeviceSynchronize());
    }
    timer->stop_store("Transpose (Finished Receive)");
  } else {
    C_t* copy_ptr;
    temp_ptr = cuFFT<T>::complex(mem_d[0]);
    copy_ptr = complex;
    if (cuda_aware) {
      send_ptr = cuFFT<T>::complex(mem_d[0]);
      recv_ptr = cuFFT<T>::complex(mem_d[1]);
    } else {
      recv_ptr = cuFFT<T>::complex(mem_h[0]);
      send_ptr = cuFFT<T>::complex(mem_h[1]);
    }

    if (!cuda_aware) {
      CUDA_CALL(cudaMemcpyAsync(send_ptr, temp_ptr, output_size_x*output_sizes_y[pidx]*output_size_z*sizeof(C_t), cudaMemcpyDeviceToHost));
      CUDA_CALL(cudaDeviceSynchronize());
    }
    timer->stop_store("Transpose (Packing)");

    timer->stop_store("Transpose (Start All2All)");
    MPI_Alltoallv(send_ptr, recvcounts.data(), rdispls.data(), MPI_BYTE, 
                  recv_ptr, sendcounts.data(), sdispls.data(), MPI_BYTE, comm);
    timer->stop_store("Transpose (Finished All2All)");

    for (int p = 0; p < pcnt; p++) {
      CUDA_CALL(cudaMemcpy2DAsync(&copy_ptr[output_start_y[p]*output_size_z], sizeof(C_t)*input_size_y*output_size_z,
          &recv_ptr[input_sizes_x[pidx]*output_start_y[p]*output_size_z], sizeof(C_t)*output_sizes_y[p]*output_size_z,
          sizeof(C_t)*output_sizes_y[p]*output_size_z, input_sizes_x[pidx], cuda_aware ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice, streams[p]));
    }
    CUDA_CALL(cudaDeviceSynchronize());
    timer->stop_store("Transpose (Finished Receive)");
  }
}

template<typename T>
void MPIcuFFT_Slab<T>::All2All_MPIType(void *complex_, bool forward) {
  using C_t = typename cuFFT<T>::C_t;
  C_t *complex = cuFFT<T>::complex(complex_);
  C_t *send_ptr, *recv_ptr, *temp_ptr;
  if (forward) {
    temp_ptr = cuFFT<T>::complex(mem_d[0]);
    if (cuda_aware) {
      recv_ptr = temp_ptr;
      send_ptr = complex;
    } else {
      recv_ptr = cuFFT<T>::complex(mem_h[0]);
      send_ptr = cuFFT<T>::complex(mem_h[1]);
    }

    if (!cuda_aware) {
      CUDA_CALL(cudaMemcpyAsync(send_ptr, complex, output_size_z*input_size_y*input_sizes_x[pidx]*sizeof(C_t), cudaMemcpyDeviceToHost));
      CUDA_CALL(cudaDeviceSynchronize());
    }
    timer->stop_store("Transpose (Packing)");

    timer->stop_store("Transpose (Start All2All)");
    MPI_Alltoallw(send_ptr, sendcounts.data(), sdispls.data(), MPI_PENCILS.data(), 
                  recv_ptr, recvcounts.data(), rdispls.data(), MPI_RECV.data(), comm);
    timer->stop_store("Transpose (Finished All2All)");

    if (!cuda_aware) {
        CUDA_CALL(cudaMemcpyAsync(temp_ptr, recv_ptr, output_size_z*output_sizes_y[pidx]*output_size_x*sizeof(C_t), cudaMemcpyHostToDevice));
        CUDA_CALL(cudaDeviceSynchronize());
    }
    timer->stop_store("Transpose (Finished Receive)");
  } else {
    C_t* copy_ptr;
    temp_ptr = cuFFT<T>::complex(mem_d[0]);
    if (cuda_aware) {
      send_ptr = cuFFT<T>::complex(mem_d[0]);
      recv_ptr = complex;
    } else {
      copy_ptr = complex;
      recv_ptr = cuFFT<T>::complex(mem_h[0]);
      send_ptr = cuFFT<T>::complex(mem_h[1]);
    }

    if (!cuda_aware) {
      CUDA_CALL(cudaMemcpyAsync(send_ptr, temp_ptr, output_size_x*output_sizes_y[pidx]*output_size_z*sizeof(C_t), cudaMemcpyDeviceToHost));
      CUDA_CALL(cudaDeviceSynchronize());
    }
    timer->stop_store("Transpose (Packing)");

    timer->stop_store("Transpose (Start All2All)");
    MPI_Alltoallw(send_ptr, recvcounts.data(), rdispls.data(), MPI_RECV.data(), 
                  recv_ptr, sendcounts.data(), sdispls.data(), MPI_PENCILS.data(), comm);
    timer->stop_store("Transpose (Finished All2All)");

    if (!cuda_aware) {
      CUDA_CALL(cudaMemcpyAsync(copy_ptr, recv_ptr, input_sizes_x[pidx]*input_size_y*output_size_z*sizeof(C_t), cudaMemcpyDeviceToHost));
      CUDA_CALL(cudaDeviceSynchronize());
    }
    timer->stop_store("Transpose (Finished Receive)");
  }
}

template<typename T>
void MPIcuFFT_Slab<T>::All2All_Communication(void *complex_, bool forward) {
  if (config.send_method == MPI_Type) 
    this->All2All_MPIType(complex_, forward);
  else 
    this->All2All_Sync(complex_, forward);
}

template<typename T> 
void MPIcuFFT_Slab<T>::execR2C(void *out, const void *in) {
  if (!initialized) 
    return;

  forward = true;

  using R_t = typename cuFFT<T>::R_t;
  using C_t = typename cuFFT<T>::C_t;
  R_t *real    = cuFFT<T>::real(in);
  C_t *complex = cuFFT<T>::complex(out);
  timer->start();
  if (fft3d) {
    CUFFT_CALL(cuFFT<T>::execR2C(planR2C, real, complex));
    CUDA_CALL(cudaDeviceSynchronize());
  } else {
    // compute 2d FFT 
    CUFFT_CALL(cuFFT<T>::execR2C(planR2C, real, complex));

    C_t *temp_ptr;
    temp_ptr = cuFFT<T>::complex(mem_d[0]);

    timer->stop_store("2D FFT (Sync)");
    CUDA_CALL(cudaDeviceSynchronize());
    timer->stop_store("2D FFT Y-Z-Direction");
  
    if (config.comm_method == Peer2Peer) 
      Peer2Peer_Communication((void *)complex);
    else if (config.comm_method == All2All)
      All2All_Communication((void *)complex);

    // compute remaining 1d FFT, for cuda-aware recv and temp buffer are identical
    CUFFT_CALL(cuFFT<T>::execC2C(planC2C, temp_ptr, complex, CUFFT_FORWARD));
    CUDA_CALL(cudaDeviceSynchronize());
    timer->stop_store("1D FFT X-Direction");

    if (config.comm_method == Peer2Peer) {
      if (config.send_method == Streams)
        mpisend_thread.join();
      MPI_Waitall(pcnt, send_req.data(), MPI_STATUSES_IGNORE);
    }
  }
  timer->stop_store("Run complete");
  if (config.warmup_rounds == 0) 
      timer->gather();
  else 
      config.warmup_rounds--;
}

template<typename T> 
void MPIcuFFT_Slab<T>::execC2R(void *out, const void *in) {
  if (!initialized) 
    return;

  forward = false;

  using R_t = typename cuFFT<T>::R_t;
  using C_t = typename cuFFT<T>::C_t;

  C_t *complex = cuFFT<T>::complex(in);
  R_t *real    = cuFFT<T>::real(out);

  timer->start();
  if (fft3d) {
    CUFFT_CALL(cuFFT<T>::execC2R(planC2R, complex, real));
    CUDA_CALL(cudaDeviceSynchronize());
  } else {
    C_t *temp_ptr, *copy_ptr;
    temp_ptr = cuFFT<T>::complex(mem_d[0]);
    copy_ptr = complex;

    // compute 1d complex to complex FFT in x direction
    CUFFT_CALL(cuFFT<T>::execC2C(planC2C, complex, temp_ptr, CUFFT_INVERSE));
    CUDA_CALL(cudaDeviceSynchronize());
    timer->stop_store("1D FFT X-Direction");

    if (config.comm_method == Peer2Peer) 
      Peer2Peer_Communication((void *)complex, false);
    else 
      All2All_Communication((void *)complex, false);

    CUFFT_CALL(cuFFT<T>::execC2R(planC2R, copy_ptr, real));
    CUDA_CALL(cudaDeviceSynchronize());
    timer->stop_store("2D FFT Y-Z-Direction");

    if (config.comm_method == Peer2Peer) {
      if (config.send_method == Streams && !cuda_aware)
        mpisend_thread.join();
      MPI_Waitall(pcnt, send_req.data(), MPI_STATUSES_IGNORE);
    }
    timer->stop_store("Run complete");
    if (config.warmup_rounds == 0) 
        timer->gather();
    else 
        config.warmup_rounds--;
  }
}

template class MPIcuFFT_Slab<float>;
template class MPIcuFFT_Slab<double>;