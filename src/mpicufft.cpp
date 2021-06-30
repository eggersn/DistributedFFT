#include "mpicufft.hpp"
#include "cufft.hpp"
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

decltype(cufftExecD2Z)* cuFFT<double>::execR2C = cufftExecD2Z;
decltype(cufftExecZ2D)* cuFFT<double>::execC2R = cufftExecZ2D;
decltype(cufftExecZ2Z)* cuFFT<double>::execC2C = cufftExecZ2Z;

template<typename T> 
MPIcuFFT<T>::MPIcuFFT(Configurations config, MPI_Comm comm, int max_world_size) : comm(comm), config(config), cuda_aware(config.cuda_aware) {
    MPI_Comm_size(comm, &pcnt);
    MPI_Comm_rank(comm, &pidx);

    if (max_world_size > 0 && pcnt > max_world_size)
        pcnt = max_world_size;

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
}

template<typename T> MPIcuFFT<T>::~MPIcuFFT() {
  if (allocated_d && workarea_d) 
    cudaFree(workarea_d);
  if (allocated_h && workarea_h) 
    cudaCheck(cudaFreeHost(workarea_h));
}

template class MPIcuFFT<float>;
template class MPIcuFFT<double>;