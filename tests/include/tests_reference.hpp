#pragma once

#include "tests_base.hpp"
#include "mpi.h"
#include "mpi-ext.h"
#include "timer.hpp"

template<typename T> 
class Tests_Reference {
public:
      Tests_Reference(size_t Nx, size_t Ny, size_t Nz, bool allow_cuda_aware, size_t P1, size_t P2) :
            Nx(Nx), Ny(Ny), Nz(Nz), P1(P1), P2(P2) {
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
      *  For a given grid \f$P_1 \times P_1\f$, each rank starts by generating random input data for \f$\frac{N_x}{P_1} \cdot \frac{N_y}{P_2} \cdot N_z\f$. 
      *  Afterwards each rank sends the data to rank 0. Here we simply wait for all input data to arrive and compute the complete 3D FFT.
      *  Finally, rank 0 distributes the computed result to the individual ranks,
      *  such that each rank ends up with the same partition \f$\frac{N_x}{P_1} \cdot \frac{N_y}{P_2} \cdot \lfloor \frac{N_z}{2}+1 \rfloor\f$.
      *
      * @param runs Specifies the number of repetitions
      */
      int testcase0(const int runs);
      /**
      *  \brief A simple reference testcase which measures the bandwidth in MB/s for each rank.
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
      int testcase2(const int opt, const int runs);
      int initializeRandArray(void* in_d, size_t N1, size_t N2);

      size_t Nx, Ny, Nz;
      size_t P1; //!< Number of partitions in x-direction
      size_t P2; //!< Number of partitions in y-direction
      bool cuda_aware = false; //!< Indicates, whether CUDA-aware MPI is available **and** selected
      Timer *timer; //!< Benchmark timer
      std::vector<std::string> section_descriptions = {"init", "Finished Send", "3D FFT", "Finished Receive", "Run complete"}; //!< Tags for the different timer sections.
};