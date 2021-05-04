#pragma once

#include <iostream>
#include <vector>
#include <mpi.h>

template<typename T> class MPIcuFFT {
public:
    MPIcuFFT (MPI_Comm comm=MPI_COMM_WORLD, bool mpi_cuda_aware=false);
    ~MPIcuFFT ();

    virtual void initFFT(size_t nx, size_t ny, size_t nz, bool allocate=true)= 0;
    virtual void setWorkArea(void *device=nullptr, void *host=nullptr) = 0;

    virtual void execR2C(void *out, const void *in) = 0;
    // virtual void execC2R(void *out, const void *in) = 0;

    virtual inline void getInSize(size_t *isize) = 0;
    virtual inline void getInStart(size_t *istart) = 0;
    virtual inline void getOutSize(size_t *osize) = 0;
    virtual inline void getOutStart(size_t *ostart) = 0;

    inline size_t getDomainSize() const { return domainsize; };
    inline size_t getWorkSizeDevice() const { return worksize_d; };
    inline size_t getWorkSizeHost() const { return worksize_h; };

    inline void* getWorkAreaDevice() const { return workarea_d; };
    inline void* getWorkAreaHost() const { return workarea_h; };

    inline int getRank() const { return pidx; };
    inline int getWorldSize() const { return pcnt; };

  
protected:  
    enum commMode_e {Peer, All2All} comm_mode;
    MPI_Comm comm;

    int pidx, pcnt;

    std::vector<int> comm_order;

    std::vector<MPI_Request> send_req;
    std::vector<MPI_Request> recv_req;

    size_t domainsize;
    size_t fft_worksize;
    
    size_t worksize_d;
    size_t worksize_h;
    
    void* workarea_d;
    void* workarea_h;

    bool allocated_d, allocated_h;
    bool cuda_aware;
    bool initialized;
    bool fft3d;
    bool half_batch;
};
