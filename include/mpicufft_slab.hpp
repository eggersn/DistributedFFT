#pragma once

#include "mpicufft.hpp"
#include "timer.hpp"
#include <cufft.h>
#include <cuda.h>
#include <vector>
#include <thread> 
#include <mutex>
#include <condition_variable>

template<typename T> class MPIcuFFT_Slab : public MPIcuFFT<T> {
public:
    MPIcuFFT_Slab (MPI_Comm comm=MPI_COMM_WORLD, bool mpi_cuda_aware=false, int max_world_size=-1);
    ~MPIcuFFT_Slab ();

    virtual void initFFT(GlobalSize *global_size, Partition *partition, bool allocate=true);
    void initFFT(GlobalSize *global_size, bool allocate=true) {
      initFFT(global_size, nullptr, allocate);
    }
    virtual void setWorkArea(void *device=nullptr, void *host=nullptr);

    virtual void execR2C(void *out, const void *in);
    // void execC2R(void *out, const void *in);

    inline void getInSize(size_t *isize) { isize[0] = input_sizes_x[pidx]; isize[1] = input_size_y; isize[2] = input_size_z; };
    inline void getInStart(size_t *istart) { istart[0] = input_start_x[pidx]; istart[1] = 0; istart[2] = 0; };
    inline void getOutSize(size_t *osize) { osize[0] = output_size_x; osize[1] = output_sizes_y[pidx]; osize[2] = output_size_z; };
    inline void getOutStart(size_t *ostart) { ostart[0] = 0; ostart[1] = output_start_y[pidx]; ostart[2] = 0; };

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

  using MPIcuFFT<T>::Peer;
  using MPIcuFFT<T>::All2All;
  using MPIcuFFT<T>::comm_mode;
  
  using MPIcuFFT<T>::comm;

  using MPIcuFFT<T>::pidx;
  using MPIcuFFT<T>::pcnt;

  using MPIcuFFT<T>::comm_order;

  using MPIcuFFT<T>::domainsize;
  using MPIcuFFT<T>::fft_worksize;

  using MPIcuFFT<T>::worksize_d;
  using MPIcuFFT<T>::worksize_h;

  using MPIcuFFT<T>::workarea_d;
  using MPIcuFFT<T>::workarea_h;

  using MPIcuFFT<T>::mem_d;
  using MPIcuFFT<T>::mem_h;

  using MPIcuFFT<T>::allocated_d;
  using MPIcuFFT<T>::allocated_h;
  using MPIcuFFT<T>::cuda_aware;
  using MPIcuFFT<T>::initialized;
  using MPIcuFFT<T>::fft3d;

  cufftHandle planR2C;
  cufftHandle planC2C;

  std::vector<size_t> input_sizes_x;
  std::vector<size_t> input_start_x;
  std::vector<size_t> output_sizes_y;
  std::vector<size_t> output_start_y;

  std::vector<MPI_Request> send_req;
  std::vector<MPI_Request> recv_req;

  std::vector<cudaStream_t> streams;

  size_t input_size_y, input_size_z;
  size_t output_size_x, output_size_z;    

  Timer *timer;

  std::vector<std::string> section_descriptions = {"init", "2D FFT Y-Z-Direction", "Transpose (First Send)", "Transpose (Packing)", "Transpose (Start Local Transpose)", 
    "Transpose (Start Receive)", "Transpose (Finished Receive)", "1D FFT X-Direction", "Run complete"};
};


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