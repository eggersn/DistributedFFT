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

  std::vector<std::string> section_descriptions = {"init", "2D FFT (Sync)", "2D FFT Y-Z-Direction", "Transpose (First Send)", "Transpose (Packing)", "Transpose (Start Local Transpose)", 
    "Transpose (Start Receive)", "Transpose (Finished Receive)", "1D FFT X-Direction", "Run complete"};
};