#pragma once

#include "mpicufft.hpp"
#include <cufft.h>
#include <cuda.h>
#include <vector>

template<typename T> class MPIcuFFT_Slabs : public MPIcuFFT<T> {
public:
    MPIcuFFT_Slabs (MPI_Comm comm=MPI_COMM_WORLD, bool mpi_cuda_aware=false);
    ~MPIcuFFT_Slabs ();

    void initFFT(size_t nx, size_t ny, size_t nz, bool allocate=true);
    void setWorkArea(void *device=nullptr, void *host=nullptr);

    void execR2C(void *out, const void *in);
    // void execC2R(void *out, const void *in);

    inline void getInSize(size_t *isize) { isize[0] = isizex[this->pidx]; isize[1] = isizey; isize[2] = isizez; };
    inline void getInStart(size_t *istart) { istart[0] = istartx[this->pidx]; istart[1] = 0; istart[2] = 0; };
    inline void getOutSize(size_t *osize) { osize[0] = osizex; osize[1] = osizey[this->pidx]; osize[2] = osizez; };
    inline void getOutStart(size_t *ostart) { ostart[0] = 0; ostart[1] = ostarty[this->pidx]; ostart[2] = 0; };

protected:
  cufftHandle planR2C;
  cufftHandle planC2R;
  cufftHandle planC2C;

  std::vector<size_t> isizex;
  std::vector<size_t> istartx;
  std::vector<size_t> osizey;
  std::vector<size_t> ostarty;
  
  std::vector<void*> mem_d;
  std::vector<void*> mem_h;

  size_t isizey, isizez;
  size_t osizex, osizez;    
};
