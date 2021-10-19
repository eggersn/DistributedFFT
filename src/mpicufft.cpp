/* 
* Copyright (C) 2021 Simon Egger
* 
* This program is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
* 
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
* 
* You should have received a copy of the GNU General Public License
* along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/

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

    if (max_world_size > 0 && pcnt > max_world_size) {
      pcnt = max_world_size;
      MPI_Comm new_comm;
      MPI_Comm_split(comm, 0, pidx, &new_comm);
      this->comm = new_comm;
    }

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