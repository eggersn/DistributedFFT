#pragma once

#include "mpicufft_pencil.hpp"

template<typename T>
class MPIcuFFT_Pencil_Opt1 : public MPIcuFFT_Pencil<T> {
public:
    MPIcuFFT_Pencil_Opt1(Configurations config, MPI_Comm comm=MPI_COMM_WORLD, int max_world_size=-1) :
        MPIcuFFT_Pencil<T>(config, comm, max_world_size) {
        planC2C_0 = 0;
    }

    ~MPIcuFFT_Pencil_Opt1();

    void initFFT(GlobalSize *global_size, Partition *partition, bool allocate=true);
    void setWorkArea(void *device=nullptr, void *host=nullptr);

    void execR2C(void *out, const void *in) { this->execR2C(out, in, 3);}
    void execR2C(void *out, const void *in, int d);

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

    void MPIsend_Thread_FirstCallback(Callback_Params_Base &params, void *ptr) {
        MPIcuFFT_Pencil<T>::MPIsend_Thread_FirstCallback_Base((void*)&params, ptr);
    }
    void MPIsend_Thread_SecondCallback(Callback_Params_Base &params, void *ptr) {
        MPIcuFFT_Pencil<T>::MPIsend_Thread_SecondCallback_Base((void*)&params, ptr);
    }

    using MPIcuFFT_Pencil<T>::config;
    using MPIcuFFT_Pencil<T>::comm;
    using MPIcuFFT_Pencil<T>::comm1;
    using MPIcuFFT_Pencil<T>::comm2;

    using MPIcuFFT_Pencil<T>::pidx;
    using MPIcuFFT_Pencil<T>::pcnt;

    using MPIcuFFT_Pencil<T>::comm_order1;
    using MPIcuFFT_Pencil<T>::comm_order2;

    using MPIcuFFT_Pencil<T>::domainsize;
    using MPIcuFFT_Pencil<T>::fft_worksize;

    using MPIcuFFT_Pencil<T>::worksize_d;
    using MPIcuFFT_Pencil<T>::worksize_h;

    using MPIcuFFT_Pencil<T>::workarea_d;
    using MPIcuFFT_Pencil<T>::workarea_h;

    using MPIcuFFT_Pencil<T>::mem_d;
    using MPIcuFFT_Pencil<T>::mem_h;

    using MPIcuFFT_Pencil<T>::allocated_d;
    using MPIcuFFT_Pencil<T>::allocated_h;
    using MPIcuFFT_Pencil<T>::cuda_aware;
    using MPIcuFFT_Pencil<T>::initialized;
    using MPIcuFFT_Pencil<T>::fft3d;

    using MPIcuFFT_Pencil<T>::global_size;
    using MPIcuFFT_Pencil<T>::partition;

    using MPIcuFFT_Pencil<T>::pidx_i;
    using MPIcuFFT_Pencil<T>::pidx_j;

    using MPIcuFFT_Pencil<T>::streams;

    using MPIcuFFT_Pencil<T>::planR2C;
    cufftHandle planC2C_0;
    using MPIcuFFT_Pencil<T>::planC2C_1;

    using MPIcuFFT_Pencil<T>::send_req;
    using MPIcuFFT_Pencil<T>::recv_req;

    using MPIcuFFT_Pencil<T>::input_dim;
    using MPIcuFFT_Pencil<T>::transposed_dim;
    using MPIcuFFT_Pencil<T>::output_dim;

    Callback_Params_Base base_params;
    std::vector<Callback_Params> params_array1;
    std::vector<Callback_Params> params_array2;

    using MPIcuFFT_Pencil<T>::timer;

    using MPIcuFFT_Pencil<T>::section_descriptions;
};