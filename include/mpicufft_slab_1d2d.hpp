#pragma once 

#include "mpicufft_slab.hpp"

template<typename T> class MPIcuFFT_Slab_1D2D : public MPIcuFFT<T> {
public:
    enum sequence_e {Z_Then_YX, Y_Then_ZX} sequence;

    MPIcuFFT_Slab_1D2D (MPI_Comm comm=MPI_COMM_WORLD, bool mpi_cuda_aware=false, int max_world_size=-1);
    ~MPIcuFFT_Slab_1D2D ();

    void initFFT(GlobalSize *global_size, Partition *partition, bool allocate=true) { initFFT(global_size, allocate); }
    void initFFT(GlobalSize *global_size, bool allocate=true, sequence_e sequence=Z_Then_YX);
    void setWorkArea(void *device=nullptr, void *host=nullptr);

    void execR2C(void *out, const void *in);

    inline void getInSize(size_t *isize) { isize[0] = input_sizes_x[pidx]; isize[1] = input_size_y; isize[2] = input_size_z; };
    inline void getInStart(size_t *istart) { istart[0] = istartx[pidx]; istart[1] = 0; istart[2] = 0; };
    inline void getOutSize(size_t *osize) { 
        osize[0] = output_size_x; 
        osize[1] = sequence == Z_Then_YX ? output_size_y : output_sizes_y[pidx]; 
        osize[2] = sequence == Z_Then_YX ? output_sizes_z[pidx] : output_size_z; 
    };
    inline void getOutStart(size_t *ostart) { 
        ostart[0] = 0;
        ostart[1] = sequence == Z_Then_YX ? 0 : ostarty[pidx]; 
        ostart[2] = sequence == Z_Then_YX ? ostartz[pidx] : 0;
    };

protected:
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

    std::vector<cufftHandle> planR2C;
    cufftHandle planC2C;

    std::vector<size_t> input_sizes_x;
    std::vector<size_t> istartx;
    std::vector<size_t> output_sizes_y;
    std::vector<size_t> ostarty;
    std::vector<size_t> output_sizes_z;
    std::vector<size_t> ostartz;

    std::vector<MPI_Request> send_req;
    std::vector<MPI_Request> recv_req;

    std::vector<cudaStream_t> streams;

    size_t input_size_y, input_size_z;
    size_t output_size_x, output_size_y, output_size_z;

    size_t ws_r2c;
    size_t num_of_streams;
};