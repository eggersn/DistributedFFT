#include "mpicufft_slab.hpp"
#include "cufft.hpp"
#include <cuda_runtime_api.h>

#if (cudaError == 0) && (cufftError == 0)
#include <stdio.h>
#include <stdlib.h>
#define cudaCheck(e) {                                           \
  int err = static_cast<int>(e);                                 \
  if(err) {                                                      \
    printf("CUDA error code %s:%d: %i\n",__FILE__,__LINE__,err); \
    exit(EXIT_FAILURE);                                          \
  }                                                              \
}
#else
#define cudaCheck(e) {e}
#endif

#define DEBUG 0
#define debug(d, v) {                                                 \
  if (DEBUG == 1) {                                                   \
    printf("DEBUG: %s: %s in %s:%d:%d\n",d,v,__FILE__,__LINE__,pidx); \
  }                                                                   \
}

#define debug_h(v) {                                                    \
  if (DEBUG == 1) {                                                   \
    printf("%s in %s:%d:%d",v,__FILE__,__LINE__,pidx);                \
  }                                                                   \
}

#define debug_int(d, v) {                                             \
  if (DEBUG == 1) {                                                   \
    printf("DEBUG: %s: %d in %s:%d:%d\n",d,v,__FILE__,__LINE__,pidx); \
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
  planC2R = 0;
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
}

template<typename T> 
MPIcuFFT_Slab<T>::~MPIcuFFT_Slab() {
    if (planR2C) 
        cudaCheck(cufftDestroy(planR2C));
    if (planC2R) 
        cudaCheck(cufftDestroy(planC2R));
    if (planC2C) 
        cudaCheck(cufftDestroy(planC2C));
}

template<typename T>
void MPIcuFFT_Slab<T>::initFFT(GlobalSize *global_size, Partition *partition, bool allocate) {
  // isizex stores how the input 3d array is distributed among the mpi processes
  size_t N1    = global_size->Nx / pcnt;
  size_t N1mod = global_size->Nx % pcnt;
  for (int p = 0; p < pcnt; ++p) {
    isizex[p]  = N1 + ((static_cast<size_t>(p) < N1mod) ? 1 : 0);
    istartx[p] = ((p==0) ? 0 : istartx[p-1]+isizex[p-1]);
  }

  //we only divide across the x-axis
  isizey = global_size->Ny; isizez = global_size->Nz;
  //if isizex[pidx] can be divided by 2, then we can overlap computation and memcpy (e.g. see execR2C)
  half_batch = (isizex[pidx]%2 == 0);
  
  //after transposing the array, it is divided across the y-axis
  size_t N2    = global_size->Ny / pcnt;
  size_t N2mod = global_size->Ny % pcnt;
  for (int p = 0; p < pcnt; ++p) {
    osizey[p]  = N2 + ((static_cast<size_t>(p) < N2mod) ? 1 : 0);
    ostarty[p] = ((p==0) ? 0 : ostarty[p-1]+osizey[p-1]);
  }
  //for real input values, the second half (of the z-axis) is symmetric to the first half
  osizex = global_size->Nx; osizez = (global_size->Nz / 2) + 1;

  debug("isizex[pdix]", std::to_string(isizex[pidx]).c_str());
  debug("osizey[pdix]", std::to_string(osizey[pidx]).c_str());
  
  if (isizex[pidx] <= 8) 
    half_batch = false;
  
  // if alltoall msg is to small ( < 512kB)
  size_t local_volume = isizex[0]*osizey[0]*osizez*sizeof(T)*2;
  if (pcnt > 4 && cuda_aware && local_volume <= 524288) 
    comm_mode = All2All;
  
  //TODO: Why the factor 2? For complex numbers?
  domainsize = 2*sizeof(T) * std::max(isizex[pidx]*isizey*((isizez/2) + 1), osizex*osizey[pidx]*osizez);
  
  //sizes of the different workspaces
  size_t ws_r2c, ws_c2r, ws_c2c;

  cudaCheck(cufftCreate(&planC2R));
  cudaCheck(cufftCreate(&planR2C));
  
  cudaCheck(cufftSetAutoAllocation(planR2C, 0));
  cudaCheck(cufftSetAutoAllocation(planC2R, 0));
  
  if (fft3d) { // combined 3d fft, in case only one mpi process is used
    cudaCheck(cufftMakePlan3d(planR2C, global_size->Nx, global_size->Ny, global_size->Nz, cuFFT<T>::R2Ctype, &ws_r2c));
    cudaCheck(cufftMakePlan3d(planC2R, global_size->Nx, global_size->Ny, global_size->Nz, cuFFT<T>::C2Rtype, &ws_c2r));

    fft_worksize = std::max(ws_r2c, ws_c2r);
  } else { // 2d slab decomposition fft
    size_t batch = (half_batch ? isizex[pidx]/2 : isizex[pidx]);
    
    //here, an additional C2C transform is needed
    cudaCheck(cufftCreate(&planC2C));
    cudaCheck(cufftSetAutoAllocation(planC2C, 0));
    
    // For the forward FFT, we start with with a 2D transform in y,z direction. Afterwards, we compute a 1D transform for the x-axis.
    long long n[3] = {static_cast<long long>(osizex), static_cast<long long>(isizey), static_cast<long long>(isizez)};
    long long nembed[1] = {1};
    
    // (1) For the forward FFT, where we can use the default data layout (thus the NULL pointer, see cuFFT doc for more details)
    // Execution order: (1) -> (3)
    cudaCheck(cufftMakePlanMany64(planR2C, 2, &n[1], 0, 0, 0, 0, 0, 0, cuFFT<T>::R2Ctype, batch, &ws_r2c));
    // (2) For the inverse FFT. Again, the default data layout can be used. 
    // Execution order: (3) -> (2)
    cudaCheck(cufftMakePlanMany64(planC2R, 2, &n[1], 0, 0, 0, 0, 0, 0, cuFFT<T>::C2Rtype, batch, &ws_c2r));
    // (3) used for the remaining 1D Transform for both (1) and (2).
    // Here, the offset of two subsequent elements (x-axis) have an offset of osizey[pidx]*osizez.
    // Assumption: Data Layout [x][y][z]
    cudaCheck(cufftMakePlanMany64(planC2C, 1, n, nembed, osizey[pidx]*osizez, 1, nembed, osizey[pidx]*osizez, 1, cuFFT<T>::C2Ctype, osizey[pidx]*osizez, &ws_c2c));
    
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

  debug("domainsize", std::to_string(domainsize).c_str());
  debug("worksize", std::to_string(fft_worksize).c_str());
  debug("worksize_d", std::to_string(worksize_d).c_str());
  debug("worksize_h", std::to_string(worksize_h).c_str());

  if (allocate) 
    this->setWorkArea();
  
  cudaCheck(cudaDeviceSynchronize());
}

//default parameters device=nullptr, host=nullptr
template<typename T> 
void MPIcuFFT_Slab<T>::setWorkArea(void *device, void *host) {
  if (!domainsize) 
    return;

  if (device && allocated_d) {
    debug("device && allocated_d", "");
    cudaCheck(cudaFree(workarea_d));
    allocated_d = false;
    workarea_d = device;
  } else if (!allocated_d && device) {
    debug("device && !allocated_d", "");
    workarea_d = device;
  } else if (!allocated_d && !device) {
    debug("!device && !allocated_d", "");
    cudaCheck(cudaMalloc(&(workarea_d), worksize_d));
    allocated_d = true;
  }

  mem_d.clear();
  for (size_t i=0; i<(worksize_d/domainsize); ++i) 
    mem_d.push_back(&static_cast<char*>(workarea_d)[i*domainsize]);

  debug("len(mem_d)", std::to_string(mem_d.size()).c_str());
  
  if (fft3d) {
    cudaCheck(cufftSetWorkArea(planR2C, mem_d[0]));
    cudaCheck(cufftSetWorkArea(planC2R, mem_d[0]));
  } else {
    cudaCheck(cufftSetWorkArea(planR2C, mem_d[2]));
    cudaCheck(cufftSetWorkArea(planC2R, mem_d[2]));
    cudaCheck(cufftSetWorkArea(planC2C, mem_d[2]));
  }
    
  if (host && allocated_h) {
    cudaCheck(cudaFreeHost(workarea_h));
    allocated_h = false;
    workarea_h = host;
  } else if (!allocated_h && host) {
    workarea_h = host;
  } else if (!allocated_h && !host && worksize_h) {
    cudaCheck(cudaMallocHost(&(workarea_h), worksize_h));
    allocated_h = true;
  }

  mem_h.clear();
  for (size_t i=0; i<(worksize_h/domainsize); ++i)
    mem_h.push_back(&static_cast<char*>(workarea_h)[i*domainsize]);

  debug("len(mem_h)", std::to_string(mem_h.size()).c_str());

  initialized = true;
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
    cudaCheck(cuFFT<T>::execR2C(planR2C, real, complex));
    cudaCheck(cudaDeviceSynchronize());
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
    cudaCheck(cuFFT<T>::execR2C(planR2C, real, complex));
    cudaCheck(cudaDeviceSynchronize());
    size_t batch = 0;

    /* half_batch is set, if isizex[pidx] % 2 == 0. Thus we compute in the above step only the first half of our batch.
    *  Afterwards we start copying the results to the send_ptr, while simultaneously computing the second half. */
    if (half_batch) {
      debug("half_batch", "true");
      batch = isizex[pidx]/2;
      for (int p=0; p<pcnt; ++p) {
        if (p == pidx) //no data needs to be send, as the receiver would be the same process
          continue;
        /* Copy the first half of the results to the corresponding part in the send buffer.
        * Importantly, even though the 2D memcpy is called, we copy the complete 3D block.
        * This is done by setting the width of the "2D array" to sizeof(C_t)*osizey[p]*osizez and the stride
        * between the columns to sizeof(C_t)*isizey*osizez*/
        size_t oslice = isizex[pidx]*osizez*ostarty[p];
        debug_h("\nhalf_batch\n");
        debug_int("p", p);
        debug_int("oslice", static_cast<int>(oslice));
        debug_int("sizeof(C_t)", sizeof(C_t));
        debug_int("batch", batch);
        debug_int("osizez", osizez);
        debug_int("osizey[p]", osizey[p]);
        debug_int("ostarty[p]", ostarty[p]);
        debug_int("isizey", isizey);
        debug_int("isizex[pidx]", isizex[pidx]);
        debug_int("dst index", oslice);
        debug_int("dpitch", sizeof(C_t)*osizey[p]*osizez);
        debug_int("src index", ostarty[p]*osizez);
        debug_int("spitch", sizeof(C_t)*isizey*osizez);
        debug_int("width", sizeof(C_t)*osizey[p]*osizez);
        debug_int("height", batch);
        debug_h("\n");
        cudaCheck(cudaMemcpy2DAsync(&send_ptr[oslice], sizeof(C_t)*osizey[p]*osizez,
                                    &complex[ostarty[p]*osizez], sizeof(C_t)*isizey*osizez,
                                    sizeof(C_t)*osizey[p]*osizez, batch,
                                    cuda_aware?cudaMemcpyDeviceToDevice:cudaMemcpyDeviceToHost));
      }
      // simultaneously compute the second half
      cudaCheck(cuFFT<T>::execR2C(planR2C, &real[isizez*isizey*batch], &complex[osizez*isizey*batch]));
      // wait until computation and memcpy synchronize
      cudaCheck(cudaDeviceSynchronize());
    } else{
      debug("half_batch", "false");
    }
    
    if (comm_mode == Peer) {
      for (auto p : comm_order) { 
        // start non-blocking receive for rank p
        MPI_Irecv((&recv_ptr[istartx[p]*osizez*osizey[pidx]]),
          sizeof(C_t)*isizex[p]*osizez*osizey[pidx], MPI_BYTE,
          p, p, comm, &(recv_req[p]));

        /* Copy results to send buffer. There are two cases. (i) batch = 0, i.e. isizex[pidx]%2==1. Then, the complete block is copied.
         * (ii) batch = isizex[pidx]/2, i.e. isizex[pidx]%2 == 0. Then half_batch == true and therefore only the second half of the block
         * has to be copied. */
        size_t oslice = isizex[pidx]*osizez*ostarty[p];
        debug_h("\ncomm\n");
        debug_int("p", p);
        debug_int("oslice", static_cast<int>(oslice));
        debug_int("sizeof(C_t)", sizeof(C_t));
        debug_int("batch", batch);
        debug_int("osizez", osizez);
        debug_int("osizey[p]", osizey[p]);
        debug_int("ostarty[p]", ostarty[p]);
        debug_int("isizey", isizey);
        debug_int("isizex[pidx]", isizex[pidx]);
        debug_int("dst index", oslice + batch*osizez*osizey[p]);
        debug_int("dpitch", sizeof(C_t)*osizey[p]*osizez);
        debug_int("src index", batch*osizez*isizey + ostarty[p]*osizez);
        debug_int("spitch", sizeof(C_t)*isizey*osizez);
        debug_int("width", sizeof(C_t)*osizey[p]*osizez);
        debug_int("height", isizex[pidx]-batch);
        debug_h("\n");
        cudaCheck(cudaMemcpy2DAsync(&send_ptr[oslice + batch*osizez*osizey[p]], sizeof(C_t)*osizey[p]*osizez,
                                    &complex[batch*osizez*isizey + ostarty[p]*osizez], sizeof(C_t)*isizey*osizez,
                                    sizeof(C_t)*osizey[p]*osizez, isizex[pidx]-batch,
                                    cuda_aware?cudaMemcpyDeviceToDevice:cudaMemcpyDeviceToHost));

        // TODO: possible optimization: avoid synchronization in each iteration
        cudaCheck(cudaDeviceSynchronize());

        // start non-blocking send
        MPI_Isend(&send_ptr[oslice], 
                  sizeof(C_t)*isizex[pidx]*osizez*osizey[p], MPI_BYTE, 
                  p, pidx, comm, &(send_req[p]));
      }
      { 
        // transpose local block
        size_t oslice = osizez*osizey[pidx]*istartx[pidx];
        debug_h("\ntranspose local block\n");
        debug_int("oslice", static_cast<int>(oslice));
        debug_int("sizeof(C_t)", sizeof(C_t));
        debug_int("osizez", osizez);
        debug_int("osizey[p]", osizey[pidx]);
        debug_int("ostarty[p]", ostarty[pidx]);
        debug_int("isizey", isizey);
        debug_int("isizex[pidx]", isizex[pidx]);
        debug_int("dst index", oslice);
        debug_int("dpitch", sizeof(C_t)*osizey[pidx]*osizez);
        debug_int("src index", ostarty[pidx]*osizez);
        debug_int("spitch", sizeof(C_t)*isizey*osizez);
        debug_int("width", sizeof(C_t)*osizey[pidx]*osizez);
        debug_int("height", isizex[pidx]);
        debug_h("\n");
        cudaCheck(cudaMemcpy2DAsync(&temp_ptr[oslice], sizeof(C_t)*osizey[pidx]*osizez,
                                    &complex[ostarty[pidx]*osizez], sizeof(C_t)*isizey*osizez, 
                                    sizeof(C_t)*osizey[pidx]*osizez, isizex[pidx],
                                    cudaMemcpyDeviceToDevice));
      }
      if (!cuda_aware) { // copy received blocks to device
        int p;
        do {
          MPI_Waitany(pcnt, recv_req.data(), &p, MPI_STATUSES_IGNORE);
          if (p == MPI_UNDEFINED) break;
          cudaCheck(cudaMemcpyAsync(&temp_ptr[istartx[p]*osizez*osizey[pidx]],
                                    &recv_ptr[istartx[p]*osizez*osizey[pidx]],
                                    isizex[p]*osizez*osizey[pidx]*sizeof(C_t), cudaMemcpyHostToDevice));
        } while(p != MPI_UNDEFINED);
      } else { // just wait for all receives
        MPI_Waitall(pcnt, recv_req.data(), MPI_STATUSES_IGNORE);
      }
      cudaCheck(cudaDeviceSynchronize());
    } else {
      std::vector<int> sendcount(pcnt, 0);
      std::vector<int> senddispl(pcnt, 0);
      std::vector<int> recvcount(pcnt, 0);
      std::vector<int> recvdispl(pcnt, 0);
      for (int p=0; p<pcnt; ++p) { // transpose each (missing) block and send it to respective process
        size_t oslice = isizex[pidx]*osizez*ostarty[p];
        sendcount[p] = sizeof(C_t)*isizex[pidx]*osizez*osizey[p];
        senddispl[p] = oslice*sizeof(C_t);
        recvcount[p] = sizeof(C_t)*isizex[p]*osizez*osizey[pidx];
        recvdispl[p] = istartx[p]*osizez*osizey[pidx]*sizeof(C_t);
        if (p == pidx) {
          cudaCheck(cudaMemcpy2DAsync(&send_ptr[oslice], sizeof(C_t)*osizey[p]*osizez,
                                      &complex[ostarty[p]*osizez], sizeof(C_t)*isizey*osizez,
                                      sizeof(C_t)*osizey[p]*osizez, isizex[pidx],
                                      cudaMemcpyDeviceToDevice));
        } else {
          cudaCheck(cudaMemcpy2DAsync(&send_ptr[oslice + batch*osizez*osizey[p]], sizeof(C_t)*osizey[p]*osizez,
                                      &complex[batch*osizez*isizey + ostarty[p]*osizez], sizeof(C_t)*isizey*osizez,
                                      sizeof(C_t)*osizey[p]*osizez, isizex[pidx]-batch,
                                      cudaMemcpyDeviceToDevice));
        }
      }
      cudaCheck(cudaDeviceSynchronize());
      MPI_Alltoallv(send_ptr, sendcount.data(), senddispl.data(), MPI_BYTE,
                    recv_ptr, recvcount.data(), recvdispl.data(), MPI_BYTE, comm);
    }
    // compute remaining 1d FFT, for cuda-aware recv and temp buffer are identical
    cudaCheck(cuFFT<T>::execC2C(planC2C, temp_ptr, complex, CUFFT_FORWARD));
    cudaCheck(cudaDeviceSynchronize());
    if (comm_mode == Peer) {
      MPI_Waitall(pcnt, send_req.data(), MPI_STATUSES_IGNORE);
    }
  }
}

template class MPIcuFFT_Slab<float>;
template class MPIcuFFT_Slab<double>;