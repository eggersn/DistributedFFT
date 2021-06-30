#pragma once

#include "tests_base.hpp"
#include "params.hpp"
#include "mpi.h"
#include "mpi-ext.h"

template<typename T> 
class Tests_Slab_Random {
public:
      virtual int run(const int testcase, const int opt, const int runs) = 0;
      void setParams(size_t Nx_, size_t Ny_, size_t Nz_, Configurations config_) {
         Nx = Nx_;
         Ny = Ny_;
         Nz = Nz_;
         config = config_;
      }

protected:
      int initializeRandArray(void* in_d, size_t N1);
      virtual  int compute(const int rank, const int world_size, const int opt, const int runs) = 0;
      virtual  int coordinate(const int world_size, const int runs) = 0;

      size_t Nx, Ny, Nz;
      Configurations config;
};
