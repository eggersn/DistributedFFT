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

#pragma once

#include "tests_base.hpp"
#include "params.hpp"
#include "mpi.h"
#include "mpi-ext.h"

template<typename T> 
class Tests_Pencil_Random {
public:
      virtual int run(const int testcase, const int opt, const int runs) = 0;
      void setParams(size_t Nx_, size_t Ny_, size_t Nz_, Configurations config_, size_t P1_, size_t P2_) {
         Nx = Nx_;
         Ny = Ny_;
         Nz = Nz_;
         P1 = P1_;
         P2 = P2_;
         config = config_;
      }

protected:
      int initializeRandArray(void* in_d, size_t N);
      virtual  int compute(const int rank, const int world_size, const int opt, const int runs) = 0;
      virtual  int coordinate(const int world_size, const int runs) = 0;

      size_t Nx, Ny, Nz;
      size_t P1, P2;
      Configurations config;
};
