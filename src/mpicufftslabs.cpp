#include "mpicufftslabs.hpp"
#include <cuda_runtime_api.h>
#include "cufft.hpp"

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
    printf("DEBUG: %s: %s in %s:%d:%d\n",d,v,__FILE__,__LINE__,this->pidx); \
  }                                                                   \
}

#define debug_h(v) {                                                    \
  if (DEBUG == 1) {                                                   \
    printf("%s in %s:%d:%d",v,__FILE__,__LINE__,this->pidx);                \
  }                                                                   \
}

#define debug_int(d, v) {                                             \
  if (DEBUG == 1) {                                                   \
    printf("DEBUG: %s: %d in %s:%d:%d\n",d,v,__FILE__,__LINE__,this->pidx); \
  }                                                                   \
}

template<typename T> 
MPIcuFFT_Slabs<T>::MPIcuFFT_Slabs(MPI_Comm comm, bool mpi_cuda_aware, int max_world_size) : MPIcuFFT<T>(comm, mpi_cuda_aware, max_world_size) {
  isizex.resize(this->pcnt, 0);
  istartx.resize(this->pcnt, 0);
  osizey.resize(this->pcnt, 0);
  ostarty.resize(this->pcnt, 0);

  isizez = 0;
  osizez = 0;

  planR2C = 0;
  planC2R = 0;
  planC2C = 0;
}

template<typename T> 
MPIcuFFT_Slabs<T>::~MPIcuFFT_Slabs() {
    if (planR2C) 
        cudaCheck(cufftDestroy(planR2C));
    if (planC2R) 
        cudaCheck(cufftDestroy(planC2R));
    if (planC2C) 
        cudaCheck(cufftDestroy(planC2C));
}

template<typename T>
void MPIcuFFT_Slabs<T>::initFFT(size_t nx, size_t ny, size_t nz, bool allocate) {
  // isizex stores how the input 3d array is distributed among the mpi processes
  size_t N1    = nx / this->pcnt;
  size_t N1mod = nx % this->pcnt;
  for (int p = 0; p < this->pcnt; ++p) {
    isizex[p]  = N1 + ((static_cast<size_t>(p) < N1mod) ? 1 : 0);
    istartx[p] = ((p==0) ? 0 : istartx[p-1]+isizex[p-1]);
  }

  //we only divide across the x-axis
  isizey = ny; isizez = nz;
  //if isizex[this->pidx] can be divided by 2, then we can overlap computation and memcpy (e.g. see execR2C)
  this->half_batch = (isizex[this->pidx]%2 == 0);
  
  //after transposing the array, it is divided across the y-axis
  size_t N2    = ny / this->pcnt;
  size_t N2mod = ny % this->pcnt;
  for (int p = 0; p < this->pcnt; ++p) {
    osizey[p]  = N2 + ((static_cast<size_t>(p) < N2mod) ? 1 : 0);
    ostarty[p] = ((p==0) ? 0 : ostarty[p-1]+osizey[p-1]);
  }
  //for real input values, the second half (of the z-axis) is symmetric to the first half
  osizex = nx; osizez = (nz / 2) + 1;

  debug("isizex[pdix]", std::to_string(isizex[this->pidx]).c_str());
  debug("osizey[pdix]", std::to_string(osizey[this->pidx]).c_str());
  
  if (isizex[this->pidx] <= 8) 
    this->half_batch = false;
  
  // if alltoall msg is to small ( < 512kB)
  size_t local_volume = isizex[0]*osizey[0]*osizez*sizeof(T)*2;
  if (this->pcnt > 4 && this->cuda_aware && local_volume <= 524288) 
    this->comm_mode = this->All2All;
  
  //TODO: Why the factor 2? For complex numbers?
  this->domainsize = 2*sizeof(T) * std::max(isizex[this->pidx]*isizey*((isizez/2) + 1), osizex*osizey[this->pidx]*osizez);
  
  //sizes of the different workspaces
  size_t ws_r2c, ws_c2r, ws_c2c;

  cudaCheck(cufftCreate(&planC2R));
  cudaCheck(cufftCreate(&planR2C));
  
  cudaCheck(cufftSetAutoAllocation(planR2C, 0));
  cudaCheck(cufftSetAutoAllocation(planC2R, 0));
  
  if (this->fft3d) { // combined 3d fft, in case only one mpi process is used
    cudaCheck(cufftMakePlan3d(planR2C, nx, ny, nz, cuFFT<T>::R2Ctype, &ws_r2c));
    cudaCheck(cufftMakePlan3d(planC2R, nx, ny, nz, cuFFT<T>::C2Rtype, &ws_c2r));

    this->fft_worksize = std::max(ws_r2c, ws_c2r);
  } else { // 2d slab decomposition fft
    size_t batch = (this->half_batch ? isizex[this->pidx]/2 : isizex[this->pidx]);
    
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
    // Here, the offset of two subsequent elements (x-axis) have an offset of osizey[this->pidx]*osizez.
    // Assumption: Data Layout [x][y][z]
    cudaCheck(cufftMakePlanMany64(planC2C, 1, n, nembed, osizey[this->pidx]*osizez, 1, nembed, osizey[this->pidx]*osizez, 1, cuFFT<T>::C2Ctype, osizey[this->pidx]*osizez, &ws_c2c));
    
    this->fft_worksize = std::max(ws_r2c, ws_c2r);
    this->fft_worksize = std::max(this->fft_worksize, ws_c2c);
  }
  
  if (this->fft_worksize < this->domainsize) 
    this->fft_worksize = this->domainsize;

  // this->worksize_d is split into 3 parts:
  // 1. space for received data, 2. space for transmitted data, 3. actual workspace (see "mem_d")
  this->worksize_d = this->fft_worksize + (this->fft3d ? 0 : 2*this->domainsize);
  // analogously for the host worksize, if mpi is not cuda-aware
  this->worksize_h = (this->cuda_aware || this->fft3d ? 0 : 2*this->domainsize);

  debug("this->domainsize", std::to_string(this->domainsize).c_str());
  debug("this->worksize", std::to_string(this->fft_worksize).c_str());
  debug("this->worksize_d", std::to_string(this->worksize_d).c_str());
  debug("this->worksize_h", std::to_string(this->worksize_h).c_str());

  if (allocate) 
    this->setWorkArea();
  
  cudaCheck(cudaDeviceSynchronize());
}

//default parameters device=nullptr, host=nullptr
template<typename T> 
void MPIcuFFT_Slabs<T>::setWorkArea(void *device, void *host) {
  if (!this->domainsize) 
    return;

  if (device && this->allocated_d) {
    debug("device && allocated_d", "");
    cudaCheck(cudaFree(this->workarea_d));
    this->allocated_d = false;
    this->workarea_d = device;
  } else if (!this->allocated_d && device) {
    debug("device && !allocated_d", "");
    this->workarea_d = device;
  } else if (!this->allocated_d && !device) {
    debug("!device && !allocated_d", "");
    cudaCheck(cudaMalloc(&(this->workarea_d), this->worksize_d));
    this->allocated_d = true;
  }

  mem_d.clear();
  for (size_t i=0; i<(this->worksize_d/this->domainsize); ++i) 
    mem_d.push_back(&static_cast<char*>(this->workarea_d)[i*this->domainsize]);

  debug("len(mem_d)", std::to_string(mem_d.size()).c_str());
  
  if (this->fft3d) {
    cudaCheck(cufftSetWorkArea(planR2C, mem_d[0]));
    cudaCheck(cufftSetWorkArea(planC2R, mem_d[0]));
  } else {
    cudaCheck(cufftSetWorkArea(planR2C, mem_d[2]));
    cudaCheck(cufftSetWorkArea(planC2R, mem_d[2]));
    cudaCheck(cufftSetWorkArea(planC2C, mem_d[2]));
  }
    
  if (host && this->allocated_h) {
    cudaCheck(cudaFreeHost(this->workarea_h));
    this->allocated_h = false;
    this->workarea_h = host;
  } else if (!this->allocated_h && host) {
    this->workarea_h = host;
  } else if (!this->allocated_h && !host && this->worksize_h) {
    cudaCheck(cudaMallocHost(&(this->workarea_h), this->worksize_h));
    this->allocated_h = true;
  }

  mem_h.clear();
  for (size_t i=0; i<(this->worksize_h/this->domainsize); ++i)
    mem_h.push_back(&static_cast<char*>(this->workarea_h)[i*this->domainsize]);

  debug("len(mem_h)", std::to_string(mem_h.size()).c_str());

  this->initialized = true;
}

template<typename T> 
void MPIcuFFT_Slabs<T>::execR2C(void *out, const void *in) {
  if (!this->initialized) 
    return;

  using R_t = typename cuFFT<T>::R_t;
  using C_t = typename cuFFT<T>::C_t;
  R_t *real    = cuFFT<T>::real(in);
  C_t *complex = cuFFT<T>::complex(out);

  if (this->fft3d) {
    cudaCheck(cuFFT<T>::execR2C(planR2C, real, complex));
    cudaCheck(cudaDeviceSynchronize());
  } else {
    C_t *recv_ptr, *send_ptr, *temp_ptr;
    temp_ptr = cuFFT<T>::complex(mem_d[0]);
    if (this->cuda_aware) {
      recv_ptr = cuFFT<T>::complex(mem_d[0]); // = temp_ptr!
      send_ptr = cuFFT<T>::complex(mem_d[1]);
    } else {
      recv_ptr = cuFFT<T>::complex(mem_h[0]);
      send_ptr = cuFFT<T>::complex(mem_h[1]);
    }
    this->recv_req[this->pidx] = MPI_REQUEST_NULL;
    this->send_req[this->pidx] = MPI_REQUEST_NULL;

    // compute 2d FFT 
    cudaCheck(cuFFT<T>::execR2C(planR2C, real, complex));
    cudaCheck(cudaDeviceSynchronize());
    size_t batch = 0;

    /* this->half_batch is set, if isizex[this->pidx] % 2 == 0. Thus we compute in the above step only the first half of our batch.
    *  Afterwards we start copying the results to the send_ptr, while simultaneously computing the second half. */
    if (this->half_batch) {
      debug("this->half_batch", "true");
      batch = isizex[this->pidx]/2;
      for (int p=0; p<this->pcnt; ++p) {
        if (p == this->pidx) //no data needs to be send, as the receiver would be the same process
          continue;
        /* Copy the first half of the results to the corresponding part in the send buffer.
        * Importantly, even though the 2D memcpy is called, we copy the complete 3D block.
        * This is done by setting the width of the "2D array" to sizeof(C_t)*osizey[p]*osizez and the stride
        * between the columns to sizeof(C_t)*isizey*osizez*/
        size_t oslice = isizex[this->pidx]*osizez*ostarty[p];
        debug_h("\nthis->half_batch\n");
        debug_int("p", p);
        debug_int("oslice", static_cast<int>(oslice));
        debug_int("sizeof(C_t)", sizeof(C_t));
        debug_int("batch", batch);
        debug_int("osizez", osizez);
        debug_int("osizey[p]", osizey[p]);
        debug_int("ostarty[p]", ostarty[p]);
        debug_int("isizey", isizey);
        debug_int("isizex[this->pidx]", isizex[this->pidx]);
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
                                    this->cuda_aware?cudaMemcpyDeviceToDevice:cudaMemcpyDeviceToHost));
      }
      // simultaneously compute the second half
      cudaCheck(cuFFT<T>::execR2C(planR2C, &real[isizez*isizey*batch], &complex[osizez*isizey*batch]));
      // wait until computation and memcpy synchronize
      cudaCheck(cudaDeviceSynchronize());
    } else{
      debug("this->half_batch", "false");
    }
    
    if (this->comm_mode == this->Peer) {
      for (auto p : this->comm_order) { 
        // start non-blocking receive for rank p
        MPI_Irecv((&recv_ptr[istartx[p]*osizez*osizey[this->pidx]]),
          sizeof(C_t)*isizex[p]*osizez*osizey[this->pidx], MPI_BYTE,
          p, p, this->comm, &(this->recv_req[p]));

        /* Copy results to send buffer. There are two cases. (i) batch = 0, i.e. isizex[this->pidx]%2==1. Then, the complete block is copied.
         * (ii) batch = isizex[this->pidx]/2, i.e. isizex[this->pidx]%2 == 0. Then this->half_batch == true and therefore only the second half of the block
         * has to be copied. */
        size_t oslice = isizex[this->pidx]*osizez*ostarty[p];
        debug_h("\ncomm\n");
        debug_int("p", p);
        debug_int("oslice", static_cast<int>(oslice));
        debug_int("sizeof(C_t)", sizeof(C_t));
        debug_int("batch", batch);
        debug_int("osizez", osizez);
        debug_int("osizey[p]", osizey[p]);
        debug_int("ostarty[p]", ostarty[p]);
        debug_int("isizey", isizey);
        debug_int("isizex[this->pidx]", isizex[this->pidx]);
        debug_int("dst index", oslice + batch*osizez*osizey[p]);
        debug_int("dpitch", sizeof(C_t)*osizey[p]*osizez);
        debug_int("src index", batch*osizez*isizey + ostarty[p]*osizez);
        debug_int("spitch", sizeof(C_t)*isizey*osizez);
        debug_int("width", sizeof(C_t)*osizey[p]*osizez);
        debug_int("height", isizex[this->pidx]-batch);
        debug_h("\n");
        cudaCheck(cudaMemcpy2DAsync(&send_ptr[oslice + batch*osizez*osizey[p]], sizeof(C_t)*osizey[p]*osizez,
                                    &complex[batch*osizez*isizey + ostarty[p]*osizez], sizeof(C_t)*isizey*osizez,
                                    sizeof(C_t)*osizey[p]*osizez, isizex[this->pidx]-batch,
                                    this->cuda_aware?cudaMemcpyDeviceToDevice:cudaMemcpyDeviceToHost));

        // TODO: possible optimization: avoid synchronization in each iteration
        cudaCheck(cudaDeviceSynchronize());

        // start non-blocking send
        MPI_Isend(&send_ptr[oslice], 
                  sizeof(C_t)*isizex[this->pidx]*osizez*osizey[p], MPI_BYTE, 
                  p, this->pidx, this->comm, &(this->send_req[p]));
      }
      { 
        // transpose local block
        size_t oslice = osizez*osizey[this->pidx]*istartx[this->pidx];
        debug_h("\ntranspose local block\n");
        debug_int("oslice", static_cast<int>(oslice));
        debug_int("sizeof(C_t)", sizeof(C_t));
        debug_int("osizez", osizez);
        debug_int("osizey[p]", osizey[this->pidx]);
        debug_int("ostarty[p]", ostarty[this->pidx]);
        debug_int("isizey", isizey);
        debug_int("isizex[this->pidx]", isizex[this->pidx]);
        debug_int("dst index", oslice);
        debug_int("dpitch", sizeof(C_t)*osizey[this->pidx]*osizez);
        debug_int("src index", ostarty[this->pidx]*osizez);
        debug_int("spitch", sizeof(C_t)*isizey*osizez);
        debug_int("width", sizeof(C_t)*osizey[this->pidx]*osizez);
        debug_int("height", isizex[this->pidx]);
        debug_h("\n");
        cudaCheck(cudaMemcpy2DAsync(&temp_ptr[oslice], sizeof(C_t)*osizey[this->pidx]*osizez,
                                    &complex[ostarty[this->pidx]*osizez], sizeof(C_t)*isizey*osizez, 
                                    sizeof(C_t)*osizey[this->pidx]*osizez, isizex[this->pidx],
                                    cudaMemcpyDeviceToDevice));
      }
      if (!this->cuda_aware) { // copy received blocks to device
        int p;
        do {
          MPI_Waitany(this->pcnt, this->recv_req.data(), &p, MPI_STATUSES_IGNORE);
          if (p == MPI_UNDEFINED) break;
          cudaCheck(cudaMemcpyAsync(&temp_ptr[istartx[p]*osizez*osizey[this->pidx]],
                                    &recv_ptr[istartx[p]*osizez*osizey[this->pidx]],
                                    isizex[p]*osizez*osizey[this->pidx]*sizeof(C_t), cudaMemcpyHostToDevice));
        } while(p != MPI_UNDEFINED);
      } else { // just wait for all receives
        MPI_Waitall(this->pcnt, this->recv_req.data(), MPI_STATUSES_IGNORE);
      }
      cudaCheck(cudaDeviceSynchronize());
    } else {
      std::vector<int> sendcount(this->pcnt, 0);
      std::vector<int> senddispl(this->pcnt, 0);
      std::vector<int> recvcount(this->pcnt, 0);
      std::vector<int> recvdispl(this->pcnt, 0);
      for (int p=0; p<this->pcnt; ++p) { // transpose each (missing) block and send it to respective process
        size_t oslice = isizex[this->pidx]*osizez*ostarty[p];
        sendcount[p] = sizeof(C_t)*isizex[this->pidx]*osizez*osizey[p];
        senddispl[p] = oslice*sizeof(C_t);
        recvcount[p] = sizeof(C_t)*isizex[p]*osizez*osizey[this->pidx];
        recvdispl[p] = istartx[p]*osizez*osizey[this->pidx]*sizeof(C_t);
        if (p == this->pidx) {
          cudaCheck(cudaMemcpy2DAsync(&send_ptr[oslice], sizeof(C_t)*osizey[p]*osizez,
                                      &complex[ostarty[p]*osizez], sizeof(C_t)*isizey*osizez,
                                      sizeof(C_t)*osizey[p]*osizez, isizex[this->pidx],
                                      cudaMemcpyDeviceToDevice));
        } else {
          cudaCheck(cudaMemcpy2DAsync(&send_ptr[oslice + batch*osizez*osizey[p]], sizeof(C_t)*osizey[p]*osizez,
                                      &complex[batch*osizez*isizey + ostarty[p]*osizez], sizeof(C_t)*isizey*osizez,
                                      sizeof(C_t)*osizey[p]*osizez, isizex[this->pidx]-batch,
                                      cudaMemcpyDeviceToDevice));
        }
      }
      cudaCheck(cudaDeviceSynchronize());
      MPI_Alltoallv(send_ptr, sendcount.data(), senddispl.data(), MPI_BYTE,
                    recv_ptr, recvcount.data(), recvdispl.data(), MPI_BYTE, this->comm);
    }
    // compute remaining 1d FFT, for cuda-aware recv and temp buffer are identical
    cudaCheck(cuFFT<T>::execC2C(planC2C, temp_ptr, complex, CUFFT_FORWARD));
    cudaCheck(cudaDeviceSynchronize());
    if (this->comm_mode == this->Peer) {
      MPI_Waitall(this->pcnt, this->send_req.data(), MPI_STATUSES_IGNORE);
    }
  }
}

template class MPIcuFFT_Slabs<float>;
template class MPIcuFFT_Slabs<double>;