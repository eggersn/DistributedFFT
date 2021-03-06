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

    void execC2R(void *out, const void *in) { this->execC2R(out, in, 3);}
    void execC2R(void *out, const void *in, int d);

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
    using MPIcuFFT_Pencil<T>::planC2R;
    cufftHandle planC2C_0_inv;
    cufftHandle planC2C_1_inv;

    using MPIcuFFT_Pencil<T>::send_req;
    using MPIcuFFT_Pencil<T>::recv_req;

    using MPIcuFFT_Pencil<T>::input_dim;
    using MPIcuFFT_Pencil<T>::transposed_dim;
    using MPIcuFFT_Pencil<T>::output_dim;

    using MPIcuFFT_Pencil<T>::timer;

    using MPIcuFFT_Pencil<T>::section_descriptions;

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

    using MPIcuFFT_Pencil<T>::forward;
};