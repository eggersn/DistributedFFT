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

template<typename T> class MPIcuFFT_Pencil : public MPIcuFFT<T> {
public:
    MPIcuFFT_Pencil (MPI_Comm comm=MPI_COMM_WORLD, bool mpi_cuda_aware=false, int max_world_size=-1);
    ~MPIcuFFT_Pencil ();

    void initFFT(GlobalSize *global_size, Partition *partition, bool allocate=true);
    void setWorkArea(void *device=nullptr, void *host=nullptr);

    void execR2C(void *out, const void *in);
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

    static void CUDART_CB MPIsend_Callback(void *data);
    void MPIsend_Thread_FirstCallback(Callback_Params_Base &params, void *ptr);
    void MPIsend_Thread_SecondCallback(Callback_Params_Base &params, void *ptr);

    void commOrder_FirstTranspose();
    void commOrder_SecondTranspose();

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

    GlobalSize *global_size;
    Partition *partition;

    size_t pidx_i;
    size_t pidx_j;

    size_t ws_c2c_0;
    size_t num_of_streams;

    std::vector<cudaStream_t> streams;

    cufftHandle planR2C;
    std::vector<cufftHandle> planC2C_0;
    cufftHandle planC2C_1;

    std::vector<MPI_Request> send_req;
    std::vector<MPI_Request> recv_req;

    Partition_Dimensions input_dim;
    Partition_Dimensions transposed_dim;
    Partition_Dimensions output_dim;

    Timer *timer;

    std::vector<std::string> section_descriptions = {"init", "1D FFT Z-Direction", "First Transpose (First Send)", "First Transpose (Packing)", "First Transpose (Start Local Transpose)", 
        "First Transpose (Start Receive)", "First Transpose (Finished Receive)", "1D FFT Y-Direction", "Second Transpose (Preparation)",
        "Second Transpose (First Send)", "Second Transpose (Packing)", "Second Transpose (Start Local Transpose)", "Second Transpose (Start Receive)", "Second Transpose (Finished Receive)", "1D FFT X-Direction", "Run complete"};
};
