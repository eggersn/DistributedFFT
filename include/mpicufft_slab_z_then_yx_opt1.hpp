#pragma once 

#include "mpicufft_slab_z_then_yx.hpp"

template<typename T> class MPIcuFFT_Slab_Z_Then_YX_Opt1 : public MPIcuFFT_Slab_Z_Then_YX<T> {
public:
    MPIcuFFT_Slab_Z_Then_YX_Opt1 (MPI_Comm comm=MPI_COMM_WORLD, bool mpi_cuda_aware=false, int max_world_size=-1) :
        MPIcuFFT_Slab_Z_Then_YX<T>(comm, mpi_cuda_aware, max_world_size) {timer->setFileName("../benchmarks/slab_z_then_yx_opt1.csv");}

    void initFFT(GlobalSize *global_size, Partition *partition, bool allocate=true) { initFFT(global_size, allocate); }
    void initFFT(GlobalSize *global_size, bool allocate=true);

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

    using MPIcuFFT_Slab_Z_Then_YX<T>::Peer;
    using MPIcuFFT_Slab_Z_Then_YX<T>::All2All;
    using MPIcuFFT_Slab_Z_Then_YX<T>::comm_mode;
    
    using MPIcuFFT_Slab_Z_Then_YX<T>::comm;

    using MPIcuFFT_Slab_Z_Then_YX<T>::pidx;
    using MPIcuFFT_Slab_Z_Then_YX<T>::pcnt;

    using MPIcuFFT_Slab_Z_Then_YX<T>::comm_order;

    using MPIcuFFT_Slab_Z_Then_YX<T>::domainsize;
    using MPIcuFFT_Slab_Z_Then_YX<T>::fft_worksize;

    using MPIcuFFT_Slab_Z_Then_YX<T>::worksize_d;
    using MPIcuFFT_Slab_Z_Then_YX<T>::worksize_h;

    using MPIcuFFT_Slab_Z_Then_YX<T>::workarea_d;
    using MPIcuFFT_Slab_Z_Then_YX<T>::workarea_h;

    using MPIcuFFT_Slab_Z_Then_YX<T>::mem_d;
    using MPIcuFFT_Slab_Z_Then_YX<T>::mem_h;

    using MPIcuFFT_Slab_Z_Then_YX<T>::allocated_d;
    using MPIcuFFT_Slab_Z_Then_YX<T>::allocated_h;
    using MPIcuFFT_Slab_Z_Then_YX<T>::cuda_aware;
    using MPIcuFFT_Slab_Z_Then_YX<T>::initialized;
    using MPIcuFFT_Slab_Z_Then_YX<T>::fft3d;

    using MPIcuFFT_Slab_Z_Then_YX<T>::planR2C;
    using MPIcuFFT_Slab_Z_Then_YX<T>::planC2C;

    using MPIcuFFT_Slab_Z_Then_YX<T>::input_sizes_x;
    using MPIcuFFT_Slab_Z_Then_YX<T>::input_start_x;
    using MPIcuFFT_Slab_Z_Then_YX<T>::output_sizes_z;
    using MPIcuFFT_Slab_Z_Then_YX<T>::output_start_z;

    using MPIcuFFT_Slab_Z_Then_YX<T>::send_req;
    using MPIcuFFT_Slab_Z_Then_YX<T>::recv_req;

    using MPIcuFFT_Slab_Z_Then_YX<T>::streams;

    using MPIcuFFT_Slab_Z_Then_YX<T>::input_size_y;
    using MPIcuFFT_Slab_Z_Then_YX<T>::input_size_z;
    using MPIcuFFT_Slab_Z_Then_YX<T>::output_size_x;
    using MPIcuFFT_Slab_Z_Then_YX<T>::output_size_y;

    using MPIcuFFT_Slab_Z_Then_YX<T>::timer;

    using MPIcuFFT_Slab_Z_Then_YX<T>::section_descriptions;
};