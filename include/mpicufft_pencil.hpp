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

#include "mpicufft.hpp"
#include "timer.hpp"
#include <cufft.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <vector>
#include <thread> 
#include <mutex>
#include <condition_variable>

/*! \page Pencil_Decomposition
    - \subpage MPIcuFFT_Pencil
    - \subpage MPIcuFFT_Pencil_Opt1
*/

/** \class MPIcuFFT_Pencil 
    \section Visualisation
    \image html graphics/Pencil.png
    The above example illustrates the procedure for P1 = P2 = 3. The pencil highlighted in green is (0, 2), i.e., \a pidx_i = 0 and \a pidx_j = 2.
    \section Details
    There are a few technical details to consider when using this option:
    - Required memory space (besides workspace required by cuFFT):
        -# If MPI is not CUDA-aware:
            - For both redistribution, an additional send- and recv-buffer (on host memory) is required
            - An additional buffer is needed, which contains the received data (on device memory) and serves as the input for the second and third FFT.
        -# If MPI is CUDA-aware:
            - Same as above, only that send- and recv-buffer are allocated on device memory
    - Required cudaMemcpy operations for each send/recv/local transpose:
        -# First redistribution:
            - send: 2D (or 3D) memcpy
            - recv: 2D (or 3D) memcpy
        -# Second redistribution:
            - send: 2D (or 3D) memcpy
            - recv: 1D memcpy (only if MPI is not CUDA-aware)
    - For the three different 1D-FFT's, we use the following cuFFT plans (with cufftMakePlanMay64)
        -# z-direction: A single plan with:
            - istride = 1, idist = Nz
            - ostride = 1, odist = \f$\lfloor Nz/2 \rfloor + 1\f$
            - batch = input_dim.size_x[pidx_i] * input_dim.size_y[pidx_j]
        -# y-direction: min(transposed_dim.size_x[pidx_i], transposed_dim.size_z[pidx_j]) plans, where each one is executed in a different stream. \n
            Each plan is defined by the following properties:
            - istride = ostride =  transposed_dim.size_z[pidx_j]
            - if transposed_dim.size_x[pidx_i] <= transposed_dim.size_z[pidx_j]: idist = odist = 1
            - else: idist = odist = transposed_dim.size_z[pidx_j] * transposed_dim.size_y[0]
            - batch = transposed_dim.size_z[pidx_j]*transposed_dim.size_x[pidx_i]
        -# x-direction: A single plan with:
            - istride = ostride = output_dim.size_y[pidx_i]*output_dim.size_z[pidx_j]
            - idist = odist = 1
            - batch = output_dim.size_z[pidx_j]*output_dim.size_y[pidx_i]
*/
template<typename T> class MPIcuFFT_Pencil : public MPIcuFFT<T> {
public:
    /** 
    * \brief Prepares for initialization (see initFFT)
    */
    MPIcuFFT_Pencil (Configurations config, MPI_Comm comm=MPI_COMM_WORLD, int max_world_size=-1);
    ~MPIcuFFT_Pencil ();

    /** 
    * \brief Creates the cuFFT plans and allocates the required memory space. 
    *
    * The function starts by initializing \a pidx_i and \a pidx_j, along with \a input_dim, \a transposed_dim and \a output_dim.
    * Afterwards, the cuFFT plans are created as described in \ref Details. Finally, setWorkArea is called to allocate the device memory.
    * 
    * @param global_size specifies the dimensions Nx, Ny and Nz (of the global input data)
    * @param partition specifies the grid size used to partition the global input data
    * @param allocate specifies if device memory has to be allocated
    */
    virtual void initFFT(GlobalSize *global_size, Partition *partition, bool allocate=true);
    /**
    * \brief Allocates the required host and device memory (see \ref Details)
    */
    virtual void setWorkArea(void *device=nullptr, void *host=nullptr);

    /**
    * \brief Computes a 3D FFT as illustrated by \ref Visualisation.
    * @param out is reused for each 1D cuFFT computation. Therefore is must hold that:
    *   - 1D FFT (z-direction): size(out) >= input_dim.size_x[pidx_i]*input_dim.size_y[pidx_j]*(Nz/2+1)
    *   - 1D FFT (y-direction): size(out) >= transposed_dim.size_x[pidx_i]*Ny*transposed_dim.size_z[pidx_j]
    *   - 1D FFT (x-direction): size(out) >= Nx*out_dim.size_y[pidx_i]*out_dim.size_z[pidx_j]
    */
    virtual void execR2C(void *out, const void *in) { this->execR2C(out, in, 3);}
    virtual void execC2R(void *out, const void *in) { this->execC2R(out, in, 3);}
    /**
    * \brief Computes the FFT in its first d dimensions as illustrated by \ref Visualisation.
    * @param out is reused for each 1D cuFFT computation. Therefore is must hold that:
    *   - 1D FFT (z-direction): size(out) >= input_dim.size_x[pidx_i]*input_dim.size_y[pidx_j]*(Nz/2+1)
    *   - 1D FFT (y-direction): size(out) >= transposed_dim.size_x[pidx_i]*Ny*transposed_dim.size_z[pidx_j]
    *   - 1D FFT (x-direction): size(out) >= Nx*out_dim.size_y[pidx_i]*out_dim.size_z[pidx_j]
    */
    virtual void execR2C(void *out, const void *in, int d);
    virtual void execC2R(void *out, const void *in, int d);
    void getPartitionDimensions(Partition_Dimensions &input_dim_, Partition_Dimensions &transposed_dim_, Partition_Dimensions &output_dim_) {
        input_dim_ = input_dim;
        transposed_dim_ = transposed_dim;
        output_dim_ = output_dim;
    }
    // void execC2R(void *out, const void *in);

    inline void getInSize(size_t *isize) { isize[0] = input_dim.size_x[pidx_i]; isize[1] = input_dim.size_y[pidx_j]; isize[2] = input_dim.size_z[0]; };
    inline void getInStart(size_t *istart) { istart[0] = input_dim.start_x[pidx_i]; istart[1] = input_dim.start_y[pidx_j]; istart[2] = 0; };
    inline void getOutSize(size_t *osize) { osize[0] = output_dim.size_x[0]; osize[1] = output_dim.size_y[pidx_i]; osize[2] = output_dim.size_z[pidx_j]; };
    inline void getOutStart(size_t *ostart) { ostart[0] = 0; ostart[1] = output_dim.start_y[pidx_i]; ostart[2] = output_dim.start_x[pidx_j]; };

protected:
    struct Callback_Params_Base {
        std::mutex mutex;
        std::condition_variable cv;
        std::vector<int> comm_ready;
    };

    struct Callback_Params {
        Callback_Params_Base *base_params;
        const size_t p;
    };

    //! \brief Called by cudaCallHostFunc after a memcpy (used by the send procedure) is complete. It notifies the sending thread that is can start sending to the corresponding rank.
    static void CUDART_CB MPIsend_Callback(void *data);

    //! \brief Sending thread for the first global redistribution
    void MPIsend_Thread_FirstCallback(Callback_Params_Base &params, void *ptr);
    //! \brief Sending thread for the second global redistribution
    void MPIsend_Thread_SecondCallback(Callback_Params_Base &params, void *ptr);

    //! \brief Can be called by MPIcuFFT_Pencil_Opt1 to avoid redundancy
    void MPIsend_Thread_FirstCallback_Base(void *data, void *ptr) {
        struct Callback_Params_Base *params = (Callback_Params_Base *) data;
        this->MPIsend_Thread_FirstCallback(*params, ptr);
    }
    //! \brief Can be called by MPIcuFFT_Pencil_Opt1 to avoid redundancy
    void MPIsend_Thread_SecondCallback_Base(void *data, void *ptr) {
        struct Callback_Params_Base *params = (Callback_Params_Base *) data;
        this->MPIsend_Thread_SecondCallback(*params, ptr);
    }

    //! \brief This method implements the Peer2Peer communication method described in \ref Communication_Methods. 
    virtual void Peer2Peer_Communication_FirstTranspose(void *complex_, bool forward=true);
    //! \brief This method implements the \a Sync (default) Peer2Peer communication method described in \ref Communication_Methods. 
    virtual void Peer2Peer_Sync_FirstTranspose(void *complex_, void *recv_ptr_, bool forward=true);
    //! \brief This method implements the \a Streams Peer2Peer communication method described in \ref Communication_Methods. 
    virtual void Peer2Peer_Streams_FirstTranspose(void *complex_, void *recv_ptr_, bool forward=true);
    //! \brief This method implements the \a MPI_Type Peer2Peer communication method described in \ref Communication_Methods. 
    virtual void Peer2Peer_MPIType_FirstTranspose(void *complex_, void *recv_ptr_, bool forward=true);
    //! \brief This method implements the All2All communication method described in \ref Communication_Methods. 
    virtual void All2All_Communication_FirstTranspose(void *complex_, bool forward=true);
    //! \brief This method implements the \a Sync (default) All2All communication method described in \ref Communication_Methods. 
    virtual void All2All_Sync_FirstTranspose(void *complex_, bool forward=true);
    //! \brief This method implements the \a MPI_Type (default) All2All communication method described in \ref Communication_Methods. 
    virtual void All2All_MPIType_FirstTranspose(void *complex_, bool forward=true);

    //! \brief This method implements the Peer2Peer communication method described in \ref Communication_Methods. 
    virtual void Peer2Peer_Communication_SecondTranspose(void *complex_, bool forward=true);
    //! \brief This method implements the \a Sync (default) Peer2Peer communication method described in \ref Communication_Methods. 
    virtual void Peer2Peer_Sync_SecondTranspose(void *complex_, void *recv_ptr_, bool forward=true);
    //! \brief This method implements the \a Streams Peer2Peer communication method described in \ref Communication_Methods. 
    virtual void Peer2Peer_Streams_SecondTranspose(void *complex_, void *recv_ptr_, bool forward=true);
    //! \brief This method implements the \a MPI_Type Peer2Peer communication method described in \ref Communication_Methods. 
    virtual void Peer2Peer_MPIType_SecondTranspose(void *complex_, void *recv_ptr_, bool forward=true);
    //! \brief This method implements the All2All communication method described in \ref Communication_Methods. 
    virtual void All2All_Communication_SecondTranspose(void *complex_, bool forward=true);
    //! \brief This method implements the \a Sync (default) All2All communication method described in \ref Communication_Methods. 
    virtual void All2All_Sync_SecondTranspose(void *complex_, bool forward=true);
    //! \brief This method implements the \a MPI_Type (default) All2All communication method described in \ref Communication_Methods. 
    virtual void All2All_MPIType_SecondTranspose(void *complex_, bool forward=true);

    using MPIcuFFT<T>::config;
    using MPIcuFFT<T>::comm;
    MPI_Comm comm1;
    MPI_Comm comm2;

    using MPIcuFFT<T>::pidx;
    using MPIcuFFT<T>::pcnt;

    std::vector<int> comm_order1;
    std::vector<int> comm_order2;

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

    GlobalSize *global_size;
    Partition *partition;

    size_t pidx_i;
    size_t pidx_j;

    size_t ws_c2c_0; 
    size_t num_of_streams;

    std::vector<cudaStream_t> streams;

    cufftHandle planR2C; 
    cufftHandle planC2R; 
    std::vector<cufftHandle> planC2C_0;
    cufftHandle planC2C_1; 

    std::vector<MPI_Request> send_req;
    std::vector<MPI_Request> recv_req;

    Partition_Dimensions input_dim;
    Partition_Dimensions transposed_dim;
    Partition_Dimensions output_dim;

    // For Peer2Peer Streams
    std::thread mpisend_thread1;
    std::thread mpisend_thread2;
    Callback_Params_Base base_params;
    std::vector<Callback_Params> params_array1;
    std::vector<Callback_Params> params_array2;

    // For MPI_Type
    std::vector<MPI_Datatype> MPI_SND1;
    std::vector<MPI_Datatype> MPI_RECV1;
    std::vector<MPI_Datatype> MPI_SND2;
    std::vector<MPI_Datatype> MPI_RECV2;

    // For All2All Communication
    std::vector<int> sendcounts1;
    std::vector<int> sdispls1;
    std::vector<int> recvcounts1;
    std::vector<int> rdispls1;

    std::vector<int> sendcounts2;
    std::vector<int> sdispls2;
    std::vector<int> recvcounts2;
    std::vector<int> rdispls2;

    Timer *timer;

    std::vector<std::string> section_descriptions = {
        "init", 
        "1D FFT Z-Direction", 
        "First Transpose (First Send)", 
        "First Transpose (Packing)", 
        "First Transpose (Start Local Transpose)",  
        "First Transpose (Start Receive)", 
        "First Transpose (First Receive)", 
        "First Transpose (Finished Receive)", 
        "First Transpose (Start All2All)", 
        "First Transpose (Finished All2All)", 
        "First Transpose (Unpacking)", 
        "First Transpose (Send Complete)", 
        "1D FFT Y-Direction", 
        "Second Transpose (First Send)", 
        "Second Transpose (Packing)", 
        "Second Transpose (Start Local Transpose)", 
        "Second Transpose (Start Receive)",
        "Second Transpose (First Receive)",
        "Second Transpose (Finished Receive)", 
        "Second Transpose (Start All2All)", 
        "Second Transpose (Finished All2All)", 
        "Second Transpose (Unpacking)", 
        "1D FFT X-Direction", 
        "Run complete"};

    bool forward = true;
};
