#pragma once 

#include "mpicufft_slab_z_then_yx.hpp"

template<typename T> class MPIcuFFT_Slab_Z_Then_YX_Opt1 : public MPIcuFFT_Slab_Z_Then_YX<T> {
public:
    MPIcuFFT_Slab_Z_Then_YX_Opt1 (Configurations config, MPI_Comm comm=MPI_COMM_WORLD, int max_world_size=-1) :
        MPIcuFFT_Slab_Z_Then_YX<T>(config, comm, max_world_size) {}

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

    void Peer2Peer_Communication(void *complex_);
    void Peer2Peer_Sync(void *complex_, void *recv_ptr_);
    void Peer2Peer_Streams(void *complex_, void *recv_ptr_);
    void Peer2Peer_MPIType(void *complex_, void *recv_ptr_);
    void All2All_Communication(void *complex_);
    void All2All_Sync(void *complex_);
    void All2All_MPIType(void *complex_);

    using MPIcuFFT_Slab_Z_Then_YX<T>::config;
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

    // For Peer2Peer Streams
    using MPIcuFFT_Slab_Z_Then_YX<T>::mpisend_thread;
    Callback_Params_Base base_params;
    std::vector<Callback_Params> params_array;

    // For MPI_Type send method
    using MPIcuFFT_Slab_Z_Then_YX<T>::MPI_PENCILS;
    std::vector<MPI_Datatype> MPI_SND;

    // For All2All Communication
    using MPIcuFFT_Slab_Z_Then_YX<T>::sendcounts;
    using MPIcuFFT_Slab_Z_Then_YX<T>::sdispls;
    using MPIcuFFT_Slab_Z_Then_YX<T>::recvcounts;
    using MPIcuFFT_Slab_Z_Then_YX<T>::rdispls;
};