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
#include "mpi.h"
#include "mpi-ext.h"
#include "timer.hpp"

template<typename T> 
class Tests_Reference {
public:
      Tests_Reference(size_t Nx, size_t Ny, size_t Nz, bool allow_cuda_aware, int warmup_rounds, std::string benchmark_dir, size_t P1, size_t P2) :
            Nx(Nx), Ny(Ny), Nz(Nz), P1(P1), P2(P2), warmup_rounds(warmup_rounds), benchmark_dir(benchmark_dir) {
            cuda_aware = allow_cuda_aware * MPIX_Query_cuda_support();
      }

      /**
      * \brief Executes the selected testcase for the specified number of repetitions
      * 
      * @param testcase It should hold \f$0 \leq testcase \leq 2\f$
      * @param opt Selects a specific option for a given testcase, in case multiple options exist. Otherwise, this parameter is disregarded.
      * @param runs Specifies the number of repetitions. The argument is passed to the selected testcase. 
      */
      int run(const int testcase, const int opt, const int runs);

protected:
      /**
      *  \brief A simple reference testcase, where the complete input is gathered on a single MPI rank to compute the complete 3D FFT with cuFFT.
      * 
      *  For a given grid \f$P_1 \times P_2\f$, each rank starts by generating random input data for \f$\frac{N_x}{P_1} \cdot \frac{N_y}{P_2} \cdot N_z\f$. 
      *  Afterwards each rank sends the data to rank 0. Here we simply wait for all input data to arrive and compute the complete 3D FFT.
      *  Finally, rank 0 distributes the computed result to the individual ranks,
      *  such that each rank ends up with the same partition \f$\frac{N_x}{P_1} \cdot \frac{N_y}{P_2} \cdot \lfloor \frac{N_z}{2}+1 \rfloor\f$.
      *
      * @param runs Specifies the number of repetitions
      */
      int testcase0(const int runs);
      /**
      *  \brief A simple reference testcase which measures the bandwidth with cudaMemcpy1D in MB/s for each rank.
      * 
      *  Each process generates a random input of size \f$N_x*N_y*N_z\f$. Afterwards the same data is send to all other ranks. If cuda_aware is disabled, the input data is copied in each *  iteration to pinned memory. There are multiple available options to consider:
      *  - opt=0: MPI uses Peer2Peer communication
      *  - opt=1: MPI uses All2All communication
      *
      * @param opt Selects one of the above options.
      * @param runs Specifies the number of iterations across which the benchmark averages the result. 
      * There are an additional 10 warm-up rounds, which are not considered for the resulting bandwidth.
      */
      int testcase1(const int opt, const int runs);
      /**
      *  \brief A simple reference testcase which measures the bandwidth with cudaMemcpy2D (for sender only) in MB/s for each rank. This testcase is mostly relevant for the slab decomposition (2D->1D).
      * 
      *  Each process generates a random input of size \f$N_{p_x}*N_y*N_z\f$. Afterwards the region is split into multiple parts across the y-axis, where the i'th region is send to rank i.
      *  Therefore the sending process has to perform a 2D-Memcpy to the send buffer. The receiving process simply gathers all arriving messages (with an 1D-Memcpy for non CUDA-aware MPI versions).
      *  There are multiple options to consider:
      *  - opt=0: Sender performs cudaMemcpy2D from device to pinned memory, such that the relevant data is continuous. Afterwards the data is send as MPI_BYTE.
      *  - opt=1: Same as opt=0, except that cudaMemcpy2D is performed on different streams and the MPI_Isend routine is called indirectly via cudaLaunchHostFunc.
      *  - opt=2: Sender performs cudaMemcpy(1D) from device to pinned memory (for non CUDA-aware MPI) and sends non-continuous data with a custom data type (via MPI_Type_vector).
      *
      * @param opt Selects one of the above options.
      * @param runs Specifies the number of iterations across which the benchmark averages the result. 
      * There are an additional 10 warm-up rounds, which are not considered for the resulting bandwidth.
      */
      int testcase2(const int opt, const int runs);
      /**
      *  \brief A simple reference testcase which measures the bandwidth with cudaMemcpy3D (for sender & receiver) in MB/s for each rank. This testcase is mostly relevant for the pencil decomposition.
      * 
      *  Each process generates a random input of size \f$N_{p_x}*N_{p_y}*N_z\f$. Afterwards the region is split into multiple parts across the z-axis, where the i'th region is send to rank i.
      *  Therefore the sending process has to perform a 3D-Memcpy to the send buffer. The receive buffer is of size \f$N_{p_x}*N_{y}*N_{p_z}\f$. Therefore, we need an additional 3D-Memcpy. 
      *  There are multiple options to consider:
      *  - opt=0: Both sender and receiver perform a 3D-Memcpy. The receiver uses MPI_Waitany to overlap the receive and copy routines.
      *  - opt=1: Same as opt=0, except that cudaMemcpy3D is performed on different streams and the MPI_Isend routine is called indirectly via cudaLaunchHostFunc.
      *  - opt=2: Both sender and receiver use custom MPI_Types (advantage: For CUDA-aware MPI no recv/send buffer has to be allocated).
      *
      * @param opt Selects one of the above options.
      * @param runs Specifies the number of iterations across which the benchmark averages the result. 
      * There are an additional 10 warm-up rounds, which are not considered for the resulting bandwidth.
      */
      int testcase3(const int opt, const int runs);
      int testcase4(const int opt, const int runs);
      int initializeRandArray(void* in_d, size_t N1, size_t N2);

      size_t Nx, Ny, Nz;
      size_t P1; //!< Number of partitions in x-direction
      size_t P2; //!< Number of partitions in y-direction
      bool cuda_aware = false; //!< Indicates, whether CUDA-aware MPI is available **and** selected
      int warmup_rounds = 0; //!< Number of rounds, before the timer is started.
      std::string benchmark_dir = "../benchmarks"; //!< benchmark directory.
      Timer *timer; //!< Benchmark timer
      std::vector<std::string> section_descriptions = {"init", "Finished Send", "3D FFT", "Finished Receive", "Run complete"}; //!< Tags for the different timer sections.
};