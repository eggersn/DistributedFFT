#pragma once

#include "mpicufft_slab.hpp"


/** \class MPIcuFFT_Slab_Opt1
    \section Visualisation
    \image html graphics/Slab_Default_Opt1.png
    The above example illustrates the procedure for P = 3. Each slab is processed by a single rank (e.g. the green slab by rank 0). At first, the FFT is computed in z- and y- direction, where the 
    coordinate system is transformed as specified by the cuFFT plan. After the global redistribution, the remaining FFT in x-direction is computed as well. Using a second transformation of the 
    coordinate system, the layout of the output data is the same as the layout of the input data.
    \section Details
    There are a few technical details to consider when using this option:
    - Required memory space (besides workspace required by cuFFT):
      -# Send Method: \a Sync or \a Streams:
        - An additional send-buffer is only required if MPI is not CUDA-aware (since the area highlighted in red is continuous).
        - An additional recv- and temp-buffer is required, since a 2D memcpy is needed in order to format the received data.
        - Depending on whether MPI is CUDA-aware, the send- and recv-buffer are allocated on device or host memory.
        - The temp-buffer is always allocated on device memory and can be used as the input for the FFT in x-direction.
      -# Send Method: \a MPI_Types:
        - Same as above, only that an additional recv-buffer can be omitted if MPI is CUDA-aware.
    - Required cudaMemcpy operations for each send/recv/local transpose:
      -# Send Method: \a Sync or \a Streams:
        - send: 1D memcpy (only if MPI is not CUDA-aware)
        - recv: 2D (or 3D) memcpy
      -# Send Method: \a MPI_Types:
        - send & recv: 1D memcpy (only if MPI is not CUDA-aware)
        - Beware: CUDA-aware MPI + \a MPI_Types might result in an enormous performance loss
    - For the two different FFT's, we use the following cuFFT plans (with cufftMakePlanMay64):
      -# zy-direction: 
        - istride = 1, inembed[1] = Nz, idist = Nz*Ny
        - ostride = input_sizes_x[pidx], onembed[1] = Nz/2+1, odist = 1
        - batch = input_sizes_x[pidx]
      -# x-direction:
        - istride = 1, idist = Nx
        - ostride = (Nz/2+1)*output_sizes_y[pidx], odist = 1
        - batch = (Nz/2+1)*output_sizes_y[pidx]
    \section Communication_Methods
    There are two available communication methods:
      -# Peer2Peer MPI Communication:
      Here, the MPI procedures \a MPI_Isend and \a MPI_Irecv are used for non-blocking communication between the different ranks. As can be seen in \ref Visualization, each rank has to receive
      a non-continuous region (highlighted in red) to rank 2. Therefore, the receiving procedure has to perform a cudaMemcpy2D before it can start computing the FFT in x-direction.
      To interleave cudaMemcpy2D with MPI_Irecv, there are two available options:
        -# \a Sync (default): Receive the data via MPI_Waitany and perform a 2D memcpy thereafter. 
        -# (\a Streams): Same as Sync, except that for non CUDA-aware MPI the copy is copied in small batches to the send buffer. 
        -# \a MPI_Type: Here, we avoid the 2D memcpy altogether and use MPI_Type_vector to receive a non-continuous data region.
      -# All2All MPI Communication:
      Here, the MPI procedures \a MPI_Alltoallv (for \a Sync) and \a MPI_Alltoallw (for \a MPI_Type) are used for global communication between all ranks. As above, there are multiple options to
      prepare the sending procedure:
        -# \a Sync (default): Copy the non-continuous regions in a seperated recv-buffer. After \a MPI_Alltoallv is complete, the data is copied to the temp-buffer in batches (each time with cudaMemcpy2D)
        -# \a MPI_Type: Again, MPI_Type_vector can be used to avoid the 2D memcpy altogether.        
*/
template<typename T> class MPIcuFFT_Slab_Opt1 : public MPIcuFFT_Slab<T> {
public: 
    MPIcuFFT_Slab_Opt1 (Configurations config, MPI_Comm comm=MPI_COMM_WORLD, int max_world_size=-1) :
      MPIcuFFT_Slab<T>(config, comm, max_world_size) {}

    ~MPIcuFFT_Slab_Opt1();
    void initFFT(GlobalSize *global_size, Partition *partition, bool allocate=true);
    void initFFT(GlobalSize *global_size, bool allocate=true) {
      this->initFFT(global_size, nullptr, allocate);
    }

    void execR2C(void *out, const void *in);
    void execC2R(void *out, const void *in);
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

  void Peer2Peer_Communication(void *complex_, bool forward=true);
  void Peer2Peer_Sync(void *complex_, void *recv_ptr_, bool forward=true);
  void Peer2Peer_Streams(void *complex_, void *recv_ptr_, bool forward=true);
  void Peer2Peer_MPIType(void *complex_, void *recv_ptr_, bool forward=true);
  void All2All_Communication(void *complex_, bool forward=true);
  void All2All_Sync(void *complex_, bool forward=true);
  void All2All_MPIType(void *complex_, bool forward=true);

  using MPIcuFFT_Slab<T>::config;  
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
  using MPIcuFFT_Slab<T>::planC2R;
  cufftHandle planC2C_inv;

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

  // For Peer2Peer Streams
  using MPIcuFFT_Slab<T>::mpisend_thread;
  Callback_Params_Base base_params;
  std::vector<Callback_Params> params_array;

  // For MPI_Type send method
  using MPIcuFFT_Slab<T>::MPI_PENCILS;
  std::vector<MPI_Datatype> MPI_SND;

  // For All2All Communication
  using MPIcuFFT_Slab<T>::sendcounts;
  using MPIcuFFT_Slab<T>::sdispls;
  using MPIcuFFT_Slab<T>::recvcounts;
  using MPIcuFFT_Slab<T>::rdispls;

  using MPIcuFFT_Slab<T>::forward;
};