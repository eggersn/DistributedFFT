#include "mpicufft.hpp"
#include <cuda.h>
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

template<typename T> 
MPIcuFFT<T>::MPIcuFFT(MPI_Comm comm, bool mpi_cuda_aware) : comm(comm), cuda_aware(mpi_cuda_aware) {
    comm_mode = Peer;

    MPI_Comm_size(comm, &pcnt);
    MPI_Comm_rank(comm, &pidx);

    send_req.resize(pcnt, MPI_REQUEST_NULL);
    recv_req.resize(pcnt, MPI_REQUEST_NULL);

    domainsize = 0;
    fft_worksize = 0;
    
    worksize_d = 0;
    worksize_h = 0;

    workarea_d = nullptr;
    workarea_h = nullptr;
  
    allocated_d = false;
    allocated_h = false;
    initialized = false;
    fft3d = (pcnt == 1);
    half_batch = false;
    
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

template<typename T> MPIcuFFT<T>::~MPIcuFFT() {
  if (allocated_d && workarea_d) 
    cudaFree(workarea_d);
  if (allocated_h && workarea_h) 
    cudaCheck(cudaFreeHost(workarea_h));
}

template class MPIcuFFT<float>;
template class MPIcuFFT<double>;