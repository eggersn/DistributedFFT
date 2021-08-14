#pragma once

#include "mpicufft.hpp"
#include "timer.hpp"
#include <cufft.h>
#include <cuda.h>
#include <vector>
#include <thread> 
#include <mutex>
#include <condition_variable>

/*! \page Slab_Decomposition
    - ZY_Then_X (default)
      -# \subpage MPIcuFFT_Slab
      -# \subpage MPIcuFFT_Slab_Opt1
    - Z_Then_YX
      -# \subpage MPIcuFFT_Slab_Z_Then_YX
      -# \subpage MPIcuFFT_Slab_Z_Then_YX_Opt1
    - Y_Then_ZX
      -# \subpage MPIcuFFT_Slab_Y_Then_ZX
*/

/** \class MPIcuFFT_Slab
    \section Visualisation
    \image html graphics/Slab_Default.png
    The above example illustrates the procedure for P = 3. Each slab is processed by a single rank (e.g. the green slab by rank 2). At first, the FFT is computed in z- and y- direction.
    After the global redistribution, the remaining FFT in x-direction is computed as well. 
    \section Details
    There are a few technical details to consider when using this option:
    - Required memory space (besides workspace required by cuFFT):
      -# Send Method: \a Sync or \a Streams:
        - An additional send- and recv-buffer is required 
        - The recv-buffer can be used as the input for the remaining FFT in x-direction
        - Depending on whether MPI is CUDA-aware, the send- and recv-buffer are allocated on device or host memory.
      -# Send Method: \a MPI_Types:
        - Same as above, only that an additional send-buffer can be omitted if MPI is CUDA-aware.
    - Required cudaMemcpy operations for each send/recv/local transpose:
      -# Send Method: \a Sync or \a Streams:
        - send: 2D (or 3D) memcpy
        - recv: 1D memcpy (only if MPI is not CUDA-aware)
      -# Send Method: \a MPI_Types:
        - send & recv: 1D memcpy (only if MPI is not CUDA-aware)
        - Beware: CUDA-aware MPI + \a MPI_Types might result in an enormous performance loss
    - For the two different FFT's, we use the following cuFFT plans (with cufftMakePlanMay64):
      -# zy-direction:
        - default data layout
        - batch: input_sizes_x[pidx]
      -# x-direction:
        - istride = output_sizes_y[pidx]*output_size_z, idist = 1
        - ostride = output_sizes_y[pidx]*output_size_z, odist = 1
        - batch = output_sizes_y[pidx]*output_size_z
    \section Communication_Methods
    There are two available communication methods:
      -# Peer2Peer MPI Communication:
      Here, the MPI procedures \a MPI_Isend and \a MPI_Irecv are used for non-blocking communication between the different ranks. As can be seen in \ref Visualization, each rank has to send
      a non-continuous region (highlighted in red) to rank 2. Therefore, the sending procedure has to perform a cudaMemcpy2D before it can start sending. To interleave cudaMemcpy2D with 
      MPI_Isend, there are three available options:
        -# \a Sync (default): Copy each non-continuous region (e.g. the one highlighted in red) and call cudaDeviceSynchronize before MPI_Isend.
        -# \a Streams: Instead of using cudaDeviceSynchronize, the sending procedure is called in a second thread. The thread is notified by cudaCallHostFunc after the relevant memcpy is complete.
        -# \a MPI_Type: Here, we avoid the 2D memcpy altogether and use MPI_Type_vector to send a non-continuous data region.
      -# All2All MPI Communication:
      Here, the MPI procedures \a MPI_Alltoallv (for \a Sync) and \a MPI_Alltoallw (for \a MPI_Type) are used for global communication between all ranks. As above, there are multiple options to
      prepare the sending procedure:
        -# \a Sync (default): Copy the non-continuous regions in a seperated send-buffer using cudaMemcpy2DAsync. Before calling \a MPI_Alltoallv, one has to call cudaDeviceSynchronize.
        -# \a MPI_Type: Again, MPI_Type_vector can be used to avoid the 2D memcpy altogether.
*/

template<typename T> class MPIcuFFT_Slab : public MPIcuFFT<T> {
public:
    /** 
    * \brief Prepares for initialization (see initFFT)
    */
    MPIcuFFT_Slab (Configurations config, MPI_Comm comm=MPI_COMM_WORLD, int max_world_size=-1);
    ~MPIcuFFT_Slab ();

    /** 
    * \brief Creates the cuFFT plans and allocates the required memory space. 
    *
    * The function starts by initializing \a pidx, along with \a input_size(s)_* and \a output_size(s)_*.
    * Afterwards, the cuFFT plans are created as described in \ref Details. Finally, setWorkArea is called to allocate the device memory.
    * 
    * @param global_size specifies the dimensions Nx, Ny and Nz (of the global input data)
    * @param partition is omitted for slab decomposition. The number of partitions is given by \a pcnt (= MPI_Comm_size).
    * @param allocate specifies if device memory has to be allocated
    */
    virtual void initFFT(GlobalSize *global_size, Partition *partition, bool allocate=true);
    void initFFT(GlobalSize *global_size, bool allocate=true) {
      initFFT(global_size, nullptr, allocate);
    }

    /**
    * \brief Allocates the required host and device memory (see \ref Details)
    */
    virtual void setWorkArea(void *device=nullptr, void *host=nullptr);

    /**
    * \brief Computes a 3D FFT as illustrated by \ref Visualisation.
    * @param out is reused for each cuFFT computation. Therefore is must hold that:
    *   - 2D FFT (zy-direction): size(out) >= input_sizes_x[pidx]*Ny*(Nz/2+1)
    *   - 1D FFT (x-direction): size(out) >= Nx*output_sizes_y[pidx]*(Nz/2+1)
    */
    virtual void execR2C(void *out, const void *in);
    virtual void execC2R(void *out, const void *in);

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

  //! \brief Only used if send_method == Streams. Then this function is called by cudaCallHostFunc to notify a second thread to start sending the copied data.
  static void CUDART_CB MPIsend_Callback(void *data);
  //! \brief Only used if send_method == Streams. Then this function is used by the sending thread for the global redistribution. 
  void MPIsend_Thread(Callback_Params_Base &params, void *ptr);

  //! \brief This method implements the Peer2Peer communication method described in \ref Communication_Methods. 
  virtual void Peer2Peer_Communication(void *complex_, bool forward=true);
  //! \brief This method implements the \a Sync (default) Peer2Peer communication method described in \ref Communication_Methods. 
  virtual void Peer2Peer_Sync(void *complex_, void *recv_ptr_, bool forward=true);
  //! \brief This method implements the \a Streams Peer2Peer communication method described in \ref Communication_Methods. 
  virtual void Peer2Peer_Streams(void *complex_, void *recv_ptr_, bool forward=true);
  //! \brief This method implements the \a MPI_Type Peer2Peer communication method described in \ref Communication_Methods. 
  virtual void Peer2Peer_MPIType(void *complex_, void *, bool forward=true);
  //! \brief This method implements the All2All communication method described in \ref Communication_Methods. 
  virtual void All2All_Communication(void *complex_, bool forward=true);
  //! \brief This method implements the \a Sync (default) All2All communication method described in \ref Communication_Methods. 
  virtual void All2All_Sync(void *complex_, bool forward=true);
  //! \brief This method implements the \a MPI_Type (default) All2All communication method described in \ref Communication_Methods. 
  virtual void All2All_MPIType(void *complex_, bool forward=true);

  using MPIcuFFT<T>::config;  
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
  cufftHandle planC2R;
  cufftHandle planC2C;

  std::vector<size_t> input_sizes_x;
  std::vector<size_t> input_start_x;
  std::vector<size_t> output_sizes_y;
  std::vector<size_t> output_start_y;

  std::vector<MPI_Request> send_req;
  std::vector<MPI_Request> recv_req;

  std::mutex send_mutex;
  std::condition_variable send_cv;
  std::unique_lock<std::mutex> *send_lk;
  bool send_complete;

  std::vector<cudaStream_t> streams;

  size_t input_size_y, input_size_z;
  size_t output_size_x, output_size_z;    

  Timer *timer;
  std::vector<std::string> section_descriptions = {
    "init", 
    "2D FFT (Sync)", 
    "2D FFT Y-Z-Direction", 
    "Transpose (First Send)", 
    "Transpose (Packing)", 
    "Transpose (Start Local Transpose)", 
    "Transpose (Start Receive)", 
    "Transpose (First Receive)", 
    "Transpose (Finished Receive)", 
    "Transpose (Start All2All)", 
    "Transpose (Finished All2All)", 
    "Transpose (Unpacking)", 
    "1D FFT X-Direction", 
    "Run complete"};

  // For Peer2Peer Streams
  std::thread mpisend_thread;
  Callback_Params_Base base_params;
  std::vector<Callback_Params> params_array;

  // For MPI_Type send method
  std::vector<MPI_Datatype> MPI_PENCILS;
  std::vector<MPI_Datatype> MPI_RECV;

  // For All2All Communication
  std::vector<int> sendcounts;
  std::vector<int> sdispls;
  std::vector<int> recvcounts;
  std::vector<int> rdispls;

  bool forward = true;
};