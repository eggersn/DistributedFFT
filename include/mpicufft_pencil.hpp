#pragma once

#include "mpicufft.hpp"
#include <cufft.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <vector>

template<typename T> class MPIcuFFT_Pencil : public MPIcuFFT<T> {
public:
    MPIcuFFT_Pencil (MPI_Comm comm=MPI_COMM_WORLD, bool mpi_cuda_aware=false, int max_world_size=-1);
    ~MPIcuFFT_Pencil ();

    void initFFT(GlobalSize *global_size, Partition *partition, bool allocate=true);
    void setWorkArea(void *device=nullptr, void *host=nullptr);

    void execR2C(void *out, const void *in);
    // void execC2R(void *out, const void *in);

    // inline void getInSize(size_t *isize) { isize[0] = input_dim.size_x[pidx]; isize[1] = isizey; isize[2] = isizez; };
    // inline void getInStart(size_t *istart) { istart[0] = istartx[pidx]; istart[1] = 0; istart[2] = 0; };
    // inline void getOutSize(size_t *osize) { osize[0] = osizex; osize[1] = osizey[pidx]; osize[2] = osizez; };
    // inline void getOutStart(size_t *ostart) { ostart[0] = 0; ostart[1] = ostarty[pidx]; ostart[2] = 0; };

protected:
    void commOrder_FirstTranspose();
    void commOrder_SecondTranspose();

    void MPIsend_Callback_FirstTranspose(cudaStream_t stream, cudaError_t status, void *data);
    void MPIsend_Callback_SecondTranspose(cudaStream_t stream, cudaError_t status, void *data);

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

    std::vector<cudaStream_t*> streams;

    cufftHandle planR2C;
    std::vector<cufftHandle*> planC2C_0;
    cufftHandle planC2C_1;

    std::vector<MPI_Request> send_req;
    std::vector<MPI_Request> recv_req;

    Partition_Dimensions input_dim;
    Partition_Dimensions transposed_dim;
    Partition_Dimensions output_dim;
};