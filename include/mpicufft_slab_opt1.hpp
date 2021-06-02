#pragma once

#include "mpicufft_slab.hpp"

template<typename T> class MPIcuFFT_Slab_Opt1 : public MPIcuFFT_Slab<T> {
public: 
    MPIcuFFT_Slab_Opt1 (MPI_Comm comm=MPI_COMM_WORLD, bool mpi_cuda_aware=false, int max_world_size=-1) :
      MPIcuFFT_Slab<T>(comm, mpi_cuda_aware, max_world_size) {timer->setFileName("../benchmarks/slab_opt1.csv");}
    void initFFT(GlobalSize *global_size, Partition *partition, bool allocate=true);
    void initFFT(GlobalSize *global_size, bool allocate=true) {
      this->initFFT(global_size, nullptr, allocate);
    }

    void execR2C(void *out, const void *in);
protected:
  struct Callback_Params_Base {
    std::mutex mutex;
    std::condition_variable cv;
    std::vector<int> comm_ready;
  };

  struct Callback_Params {
    Callback_Params_Base *base_params;
    const int p;
  };

  static void CUDART_CB MPIsend_Callback(void *data);
  void MPIsend_Thread(Callback_Params_Base &params, void *ptr);

  using MPIcuFFT_Slab<T>::Peer;
  using MPIcuFFT_Slab<T>::All2All;
  using MPIcuFFT_Slab<T>::comm_mode;
  
  using MPIcuFFT_Slab<T>::comm;

  using MPIcuFFT_Slab<T>::pidx;
  using MPIcuFFT_Slab<T>::pcnt;

  using MPIcuFFT_Slab<T>::comm_order;

  using MPIcuFFT_Slab<T>::domainsize;
  using MPIcuFFT_Slab<T>::fft_worksize;

  using MPIcuFFT_Slab<T>::worksize_d;
  using MPIcuFFT_Slab<T>::worksize_h;

  using MPIcuFFT_Slab<T>::workarea_d;
  using MPIcuFFT_Slab<T>::workarea_h;

  using MPIcuFFT_Slab<T>::mem_d;
  using MPIcuFFT_Slab<T>::mem_h;

  using MPIcuFFT_Slab<T>::allocated_d;
  using MPIcuFFT_Slab<T>::allocated_h;
  using MPIcuFFT_Slab<T>::cuda_aware;
  using MPIcuFFT_Slab<T>::initialized;
  using MPIcuFFT_Slab<T>::fft3d;

  using MPIcuFFT_Slab<T>::planR2C;
  using MPIcuFFT_Slab<T>::planC2C;

  using MPIcuFFT_Slab<T>::input_sizes_x;
  using MPIcuFFT_Slab<T>::input_start_x;
  using MPIcuFFT_Slab<T>::output_sizes_y;
  using MPIcuFFT_Slab<T>::output_start_y;

  using MPIcuFFT_Slab<T>::send_req;
  using MPIcuFFT_Slab<T>::recv_req;

  using MPIcuFFT_Slab<T>::streams;

  using MPIcuFFT_Slab<T>::input_size_y; 
  using MPIcuFFT_Slab<T>::input_size_z;
  using MPIcuFFT_Slab<T>::output_size_x;
  using MPIcuFFT_Slab<T>::output_size_z;    

  using MPIcuFFT_Slab<T>::timer;

  using MPIcuFFT_Slab<T>::section_descriptions;
};