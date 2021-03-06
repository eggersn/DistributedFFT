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

#include "mpicufft_slab_z_then_yx.hpp"
#include "cufft.hpp"
#include <cuda_runtime_api.h>


#define CUDA_CALL(x) do { if((x)!=cudaSuccess) {        \
    printf("Error %d at %s:%d\n",x,__FILE__,__LINE__);  \
    exit(EXIT_FAILURE); }} while(0)

#define CUFFT_CALL(x) do { if((x)!=CUFFT_SUCCESS) {     \
    printf("Error %d at %s:%d\n",x,__FILE__,__LINE__);  \
    exit(EXIT_FAILURE); }} while(0)

template<typename T> 
MPIcuFFT_Slab_Z_Then_YX<T>::MPIcuFFT_Slab_Z_Then_YX(Configurations config, MPI_Comm comm, int max_world_size) 
    : MPIcuFFT<T>(config, comm, max_world_size) {
    input_sizes_x.resize(pcnt, 0);
    input_start_x.resize(pcnt, 0);
    output_sizes_z.resize(pcnt, 0);
    output_start_z.resize(pcnt, 0);

    send_req.resize(pcnt, MPI_REQUEST_NULL);
    recv_req.resize(pcnt, MPI_REQUEST_NULL);

    input_size_z = 0;
    input_size_y = 0;
    output_size_y = 0;
    output_size_x = 0;

    planR2C = 0;
    planC2C = 0;
    planC2R = 0;

    for (int i = 1; i < pcnt; i++)
        comm_order.push_back((pidx + i) % pcnt);

    for (int i = 0; i < pcnt; i++){
        cudaStream_t stream;
        CUDA_CALL(cudaStreamCreate(&stream));
        streams.push_back(stream);
    }
}

template<typename T> 
MPIcuFFT_Slab_Z_Then_YX<T>::~MPIcuFFT_Slab_Z_Then_YX() {
    if (planR2C)
        CUFFT_CALL(cufftDestroy(planR2C));
    if (planC2C) 
        CUFFT_CALL(cufftDestroy(planC2C));
    if (planC2R) 
        CUFFT_CALL(cufftDestroy(planC2R));
    
    delete timer;
}


template<typename T> 
void MPIcuFFT_Slab_Z_Then_YX<T>::initFFT(GlobalSize *global_size, bool allocate) {
    mkdir((config.benchmark_dir +  "/slab_z_then_yx").c_str(), 0777);
    std::stringstream ss;
    ss << config.benchmark_dir <<  "/slab_z_then_yx/test_0_" << config.comm_method << "_" << config.send_method << "_" << global_size->Nx << "_" << global_size->Ny << "_" << global_size->Nz;
    ss << "_" << cuda_aware << "_" << pcnt << ".csv";
    std::string filename = ss.str();
    timer = new Timer(comm, 0, pcnt, pidx, section_descriptions, filename);
    timer->start();
    using R_t = typename cuFFT<T>::R_t;
    using C_t = typename cuFFT<T>::C_t;

    // input_sizes_x stores how the input 3d array is distributed among the mpi processes
    size_t N1    = global_size->Nx / pcnt;
    size_t N1mod = global_size->Nx % pcnt;
    for (int p = 0; p < pcnt; ++p) {
        input_sizes_x[p]  = N1 + ((static_cast<size_t>(p) < N1mod) ? 1 : 0);
        input_start_x[p] = ((p==0) ? 0 : input_start_x[p-1]+input_sizes_x[p-1]);
    }

    // We only divide across the x-axis
    input_size_y = global_size->Ny; input_size_z = global_size->Nz;

    // After transposing the array, it is divided across the z-axis
    size_t N2 = (global_size->Nz/2+1) / pcnt;
    size_t N2mod = (global_size->Nz/2+1) % pcnt;

    for (int p = 0; p < pcnt; p++){
        output_sizes_z[p] = N2 + ((static_cast<size_t>(p) < N2mod) ? 1 : 0);
        output_start_z[p] = ((p==0) ? 0 : output_start_z[p-1]+output_sizes_z[p-1]);
    }
    // For real input values, the second half (of the z-axis) is symmetric (complex conjugate) to the first half
    output_size_x = global_size->Nx; output_size_y = global_size->Ny; output_size_z = input_size_z/2 + 1;

    domainsize = sizeof(C_t) * std::max(input_sizes_x[pidx]*input_size_y*output_size_z, 
        output_size_x*output_size_y*output_sizes_z[pidx]);

    // Sizes of the different workspaces
    size_t ws_r2c, ws_c2c, ws_c2r;
    CUFFT_CALL(cufftCreate(&planR2C));
    CUFFT_CALL(cufftSetAutoAllocation(planR2C, 0));

    CUFFT_CALL(cufftCreate(&planC2R));
    CUFFT_CALL(cufftSetAutoAllocation(planC2R, 0));
        
    if (fft3d) { // Combined 3d fft, in case only one mpi process is used
        CUFFT_CALL(cufftMakePlan3d(planR2C, global_size->Nx, global_size->Ny, global_size->Nz, cuFFT<T>::R2Ctype, &ws_r2c));
        CUFFT_CALL(cufftMakePlan3d(planC2R, global_size->Nx, global_size->Ny, global_size->Nz, cuFFT<T>::C2Rtype, &ws_c2r));

        fft_worksize = std::max(ws_r2c, ws_c2r);
    } else {
        size_t batch = input_size_y * input_sizes_x[pidx];

        CUFFT_CALL(cufftCreate(&planC2C));
        CUFFT_CALL(cufftSetAutoAllocation(planC2C, 0));

        long long n[3] = {static_cast<long long>(output_size_x), static_cast<long long>(output_size_y), 
            static_cast<long long>(input_size_z)};

        // For the 1D R2C FFT, the default data layer can be used (in case sequence = Z_Then_YX)
        CUFFT_CALL(cufftMakePlanMany64(planR2C, 1, &n[2], 0, 0, 0, 0, 0, 0, cuFFT<T>::R2Ctype, batch, &ws_r2c));
        CUFFT_CALL(cufftMakePlanMany64(planC2R, 1, &n[2], 0, 0, 0, 0, 0, 0, cuFFT<T>::C2Rtype, batch, &ws_c2r));

        batch = output_sizes_z[pidx];
        long long nembed[2] = {1, static_cast<long long>(output_size_y)};

        CUFFT_CALL(cufftMakePlanMany64(planC2C, 2, &n[0], nembed, output_sizes_z[pidx], 1, nembed, 
            output_sizes_z[pidx], 1, cuFFT<T>::C2Ctype, batch, &ws_c2c));


        fft_worksize = std::max(std::max(ws_r2c, ws_c2c), ws_c2r);
    } 

    if (fft_worksize < domainsize)
        fft_worksize = domainsize;

    // worksize_d is split into 3 parts:
    // 1. space for received data, 2. space for transmitted data, 3. actual workspace (see "mem_d")
    worksize_d = fft_worksize + (fft3d ? 0 : (config.send_method == MPI_Type || !cuda_aware ? domainsize : 2*domainsize));
    // analogously for the host worksize, if mpi is not cuda-aware
    worksize_h = (cuda_aware || fft3d ? 0 : 2*domainsize);

    if (allocate) 
        this->setWorkArea();
    
    if (config.comm_method == Peer2Peer) {
        if (config.send_method == Streams) {
            for (int i = 0; i < pcnt; i++){
                Callback_Params params = {&base_params, i};
                params_array.push_back(params);
            }
        } else if (config.send_method == MPI_Type) {
            MPI_PENCILS = std::vector<MPI_Datatype>(pcnt);
            for (int i = 0; i < pcnt; i++) {
                MPI_Type_vector(input_sizes_x[pidx]*input_size_y, output_sizes_z[i]*sizeof(C_t), output_size_z*sizeof(C_t), MPI_BYTE, &MPI_PENCILS[i]);
                MPI_Type_commit(&MPI_PENCILS[i]);
            }
        }
    } else {
        if (config.send_method == MPI_Type) {
            MPI_PENCILS = std::vector<MPI_Datatype>(pcnt);
            MPI_RECV = std::vector<MPI_Datatype>(pcnt);

            sendcounts = std::vector<int>(pcnt, 1);
            sdispls = std::vector<int>(pcnt, 0);
            recvcounts = std::vector<int>(pcnt, 0);
            rdispls = std::vector<int>(pcnt, 0);
            for (int p = 0; p < pcnt; p++) {
                sdispls[p] = output_start_z[p]*sizeof(C_t);
                recvcounts[p] = output_sizes_z[pidx]*input_size_y*input_sizes_x[p]*sizeof(C_t);
                rdispls[p] = output_sizes_z[pidx]*input_size_y*input_start_x[p]*sizeof(C_t);
                MPI_Type_vector(input_sizes_x[pidx]*input_size_y, output_sizes_z[p]*sizeof(C_t), output_size_z*sizeof(C_t), MPI_BYTE, &MPI_PENCILS[p]);
                MPI_Type_commit(&MPI_PENCILS[p]);
                MPI_RECV[p] = MPI_BYTE;
            }
        } else {
            sendcounts = std::vector<int>(pcnt, 0);
            sdispls = std::vector<int>(pcnt, 0);
            recvcounts = std::vector<int>(pcnt, 0);
            rdispls = std::vector<int>(pcnt, 0);
            for (int p = 0; p < pcnt; p++) {
                sendcounts[p] = output_sizes_z[p]*input_size_y*input_sizes_x[pidx]*sizeof(C_t);
                sdispls[p] = output_start_z[p]*input_size_y*input_sizes_x[pidx]*sizeof(C_t);
                recvcounts[p] = output_sizes_z[pidx]*input_size_y*input_sizes_x[p]*sizeof(C_t);
                rdispls[p] = output_sizes_z[pidx]*input_size_y*input_start_x[p]*sizeof(C_t);
            }
        }
    }

    CUDA_CALL(cudaDeviceSynchronize());
    timer->stop_store("init");
}


//default parameters device=nullptr, host=nullptr
template<typename T> 
void MPIcuFFT_Slab_Z_Then_YX<T>::setWorkArea(void *device, void *host) {
    if (!domainsize) 
        return;

    if (device && allocated_d) {
        CUDA_CALL(cudaFree(workarea_d));
        allocated_d = false;
        workarea_d = device;
    } else if (!allocated_d && device) {
        workarea_d = device;
    } else if (!allocated_d && !device) {
        CUDA_CALL(cudaMalloc(&(workarea_d), worksize_d));
        allocated_d = true;
    }

    mem_d.clear();
    for (size_t i=0; i< 1 + (fft3d ? 0 : (!cuda_aware || config.send_method == MPI_Type ? 1 : 2)); ++i) 
        mem_d.push_back(&static_cast<char*>(workarea_d)[i*domainsize]);
    
    if (fft3d) {
        CUFFT_CALL(cufftSetWorkArea(planR2C, mem_d[0]));
        CUFFT_CALL(cufftSetWorkArea(planC2R, mem_d[0]));
    } else {
        CUFFT_CALL(cufftSetWorkArea(planR2C, mem_d[!cuda_aware || config.send_method == MPI_Type ? 1 : 2]));
        CUFFT_CALL(cufftSetWorkArea(planC2C, mem_d[!cuda_aware || config.send_method == MPI_Type ? 1 : 2]));
        CUFFT_CALL(cufftSetWorkArea(planC2R, mem_d[!cuda_aware || config.send_method == MPI_Type ? 1 : 2]));
    } 
        
    if (host && allocated_h) {
        CUDA_CALL(cudaFreeHost(workarea_h));
        allocated_h = false;
        workarea_h = host;
    } else if (!allocated_h && host) {
        workarea_h = host;
    } else if (!allocated_h && !host && worksize_h) {
        CUDA_CALL(cudaMallocHost(&(workarea_h), worksize_h));
        allocated_h = true;
    }

    mem_h.clear();
    for (size_t i=0; i<((fft3d || cuda_aware) ? 0 : 2); ++i)
        mem_h.push_back(&static_cast<char*>(workarea_h)[i*domainsize]);

    initialized = true;
}

template<typename T>
void MPIcuFFT_Slab_Z_Then_YX<T>::Peer2Peer_Sync(void *complex_, void *recv_ptr_, bool forward) {
    using C_t = typename cuFFT<T>::C_t;
    C_t *complex = cuFFT<T>::complex(complex_);
    C_t *recv_ptr = cuFFT<T>::complex(recv_ptr_);
    C_t *send_ptr;

    if (forward) {
        if (cuda_aware)
            send_ptr = cuFFT<T>::complex(mem_d[1]);
        else 
            send_ptr = cuFFT<T>::complex(mem_h[1]);

        for (auto p : comm_order) {
            MPI_Irecv(&recv_ptr[output_sizes_z[pidx]*output_size_y*input_start_x[p]], 
                sizeof(C_t)*output_sizes_z[pidx]*output_size_y*input_sizes_x[p], MPI_BYTE, p, p, comm, &recv_req[p]);

            size_t oslice = output_start_z[p]*input_size_y*input_sizes_x[pidx];

            CUDA_CALL(cudaMemcpy2DAsync(&send_ptr[oslice], sizeof(C_t)*output_sizes_z[p],
                &complex[output_start_z[p]], sizeof(C_t)*output_size_z, 
                sizeof(C_t)*output_sizes_z[p], input_size_y*input_sizes_x[pidx],
                cuda_aware?cudaMemcpyDeviceToDevice:cudaMemcpyDeviceToHost, streams[p]));
            CUDA_CALL(cudaDeviceSynchronize());

            if (p == comm_order[0])
                timer->stop_store("Transpose (First Send)");

            MPI_Isend(&send_ptr[oslice], sizeof(C_t)*output_sizes_z[p]*input_size_y*input_sizes_x[pidx], 
                MPI_BYTE, p, pidx, comm, &send_req[p]);
        }
        timer->stop_store("Transpose (Packing)");
    } else {
        C_t *temp_ptr = cuFFT<T>::complex(mem_d[0]);
        if (cuda_aware) 
            send_ptr = temp_ptr;
        else 
            send_ptr = cuFFT<T>::complex(mem_h[1]);

        if (!cuda_aware) {
            CUDA_CALL(cudaMemcpyAsync(send_ptr, temp_ptr, output_size_x*output_size_y*output_sizes_z[pidx]*sizeof(C_t), cudaMemcpyDeviceToHost));
            CUDA_CALL(cudaDeviceSynchronize());
        }
        timer->stop_store("Transpose (Packing)");

        for (auto p : comm_order) {
            MPI_Irecv(&recv_ptr[output_start_z[p]*input_size_y*input_sizes_x[pidx]], 
                sizeof(C_t)*output_sizes_z[p]*input_size_y*input_sizes_x[pidx], MPI_BYTE, p, p, comm, &recv_req[p]);

            if (p == comm_order[0])
                timer->stop_store("Transpose (First Send)");  

            MPI_Isend(&send_ptr[output_sizes_z[pidx]*output_size_y*input_start_x[p]], 
                sizeof(C_t)*output_sizes_z[pidx]*output_size_y*input_sizes_x[p], MPI_BYTE, p, pidx, comm, &send_req[p]);
        }
    }
}

template<typename T> 
void MPIcuFFT_Slab_Z_Then_YX<T>::MPIsend_Callback(void *data) {
  struct Callback_Params *params = (Callback_Params *)data;
  struct Callback_Params_Base *base_params = params->base_params;
  {
    std::lock_guard<std::mutex> lk(base_params->mutex);
    base_params->comm_ready.push_back(params->p);
  }
  base_params->cv.notify_one();
}

template<typename T>
void MPIcuFFT_Slab_Z_Then_YX<T>::MPIsend_Thread(Callback_Params_Base &base_params, void *ptr) {
  using C_t = typename cuFFT<T>::C_t;
  C_t *send_ptr = (C_t *) ptr;

  for (int i = 0; i < comm_order.size(); i++){
    std::unique_lock<std::mutex> lk(base_params.mutex);
    base_params.cv.wait(lk, [&base_params]{return !base_params.comm_ready.empty();});

    int p = base_params.comm_ready.back();
    base_params.comm_ready.pop_back();

    if (i == 0)
      timer->stop_store("Transpose (First Send)");

    if (forward) {
        size_t oslice = output_start_z[p]*input_size_y*input_sizes_x[pidx];
        MPI_Isend(&send_ptr[oslice], sizeof(C_t)*output_sizes_z[p]*input_size_y*input_sizes_x[pidx], 
            MPI_BYTE, p, pidx, comm, &send_req[p]);
    } else {
        MPI_Isend(&send_ptr[output_sizes_z[pidx]*output_size_y*input_start_x[p]], 
            sizeof(C_t)*output_sizes_z[pidx]*output_size_y*input_sizes_x[p], MPI_BYTE, p, pidx, comm, &send_req[p]);
    }

    lk.unlock();
  }
  timer->stop_store("Transpose (Packing)");
}

template<typename T>
void MPIcuFFT_Slab_Z_Then_YX<T>::Peer2Peer_Streams(void *complex_, void *recv_ptr_, bool forward) {
    using C_t = typename cuFFT<T>::C_t;
    C_t *complex = cuFFT<T>::complex(complex_);
    C_t *recv_ptr = cuFFT<T>::complex(recv_ptr_);
    C_t *send_ptr;

    if (forward) {
        if (cuda_aware)
            send_ptr = cuFFT<T>::complex(mem_d[1]);
        else 
            send_ptr = cuFFT<T>::complex(mem_h[1]);

        mpisend_thread = std::thread(&MPIcuFFT_Slab_Z_Then_YX<T>::MPIsend_Thread, this, std::ref(base_params), send_ptr);
        for (auto p : comm_order) {
            MPI_Irecv(&recv_ptr[output_sizes_z[pidx]*output_size_y*input_start_x[p]], 
                sizeof(C_t)*output_sizes_z[pidx]*output_size_y*input_sizes_x[p], MPI_BYTE, p, p, comm, &recv_req[p]);

            size_t oslice = output_start_z[p]*input_size_y*input_sizes_x[pidx];

            CUDA_CALL(cudaMemcpy2DAsync(&send_ptr[oslice], sizeof(C_t)*output_sizes_z[p],
                &complex[output_start_z[p]], sizeof(C_t)*output_size_z, 
                sizeof(C_t)*output_sizes_z[p], input_size_y*input_sizes_x[pidx],
                cuda_aware?cudaMemcpyDeviceToDevice:cudaMemcpyDeviceToHost, streams[p]));

            // Callback function for the specific stream
            CUDA_CALL(cudaLaunchHostFunc(streams[p], this->MPIsend_Callback, (void *)&params_array[p]));
        }
    } else {
        C_t *temp_ptr = cuFFT<T>::complex(mem_d[0]);
        if (cuda_aware) 
            send_ptr = temp_ptr;
        else 
            send_ptr = cuFFT<T>::complex(mem_h[1]);

        if (!cuda_aware) 
            mpisend_thread = std::thread(&MPIcuFFT_Slab_Z_Then_YX<T>::MPIsend_Thread, this, std::ref(base_params), send_ptr);
        else 
            timer->stop_store("Transpose (Packing)");  

        for (auto p : comm_order) {
            MPI_Irecv(&recv_ptr[output_start_z[p]*input_size_y*input_sizes_x[pidx]], 
                sizeof(C_t)*output_sizes_z[p]*input_size_y*input_sizes_x[pidx], MPI_BYTE, p, p, comm, &recv_req[p]);

            if (!cuda_aware) {
                CUDA_CALL(cudaMemcpyAsync(&send_ptr[output_sizes_z[pidx]*output_size_y*input_start_x[p]], 
                    &temp_ptr[output_sizes_z[pidx]*output_size_y*input_start_x[p]], 
                    output_sizes_z[pidx]*output_size_y*input_sizes_x[p]*sizeof(C_t), cudaMemcpyDeviceToHost, streams[p]));

                CUDA_CALL(cudaLaunchHostFunc(streams[p], this->MPIsend_Callback, (void *)&params_array[p]));
            } else {
                if (p == comm_order[0])
                    timer->stop_store("Transpose (First Send)");  

                MPI_Isend(&send_ptr[output_sizes_z[pidx]*output_size_y*input_start_x[p]], 
                    sizeof(C_t)*output_sizes_z[pidx]*output_size_y*input_sizes_x[p], MPI_BYTE, p, pidx, comm, &send_req[p]);
            }
        }
    }
}

template<typename T>
void MPIcuFFT_Slab_Z_Then_YX<T>::Peer2Peer_MPIType(void *complex_, void *recv_ptr_, bool forward) {
    using C_t = typename cuFFT<T>::C_t;
    C_t *complex = cuFFT<T>::complex(complex_);
    C_t *recv_ptr = cuFFT<T>::complex(recv_ptr_);
    C_t *send_ptr;

    if (forward) {
        if (cuda_aware)
            send_ptr = complex;
        else 
            send_ptr = cuFFT<T>::complex(mem_h[1]);

        if (!cuda_aware) {
            CUDA_CALL(cudaMemcpyAsync(send_ptr, complex, output_size_z*input_size_y*input_sizes_x[pidx]*sizeof(C_t), cudaMemcpyDeviceToHost));
            CUDA_CALL(cudaDeviceSynchronize());
        }
        timer->stop_store("Transpose (Packing)");

        for (auto p : comm_order) { 
            // start non-blocking receive for rank p
            MPI_Irecv(&recv_ptr[output_sizes_z[pidx]*output_size_y*input_start_x[p]], 
                sizeof(C_t)*output_sizes_z[pidx]*output_size_y*input_sizes_x[p], MPI_BYTE, p, p, comm, &recv_req[p]);
            if (p == comm_order[0])
                timer->stop_store("Transpose (First Send)");
            MPI_Isend(&send_ptr[output_start_z[p]], 1, MPI_PENCILS[p], p, pidx, comm, &send_req[p]);    
        }
    } else {
        C_t *temp_ptr = cuFFT<T>::complex(mem_d[0]); 
        if (cuda_aware)
            send_ptr = temp_ptr;
        else 
            send_ptr = cuFFT<T>::complex(mem_h[1]);

        if (!cuda_aware) {
            CUDA_CALL(cudaMemcpyAsync(send_ptr, temp_ptr, output_size_x*output_size_y*output_sizes_z[pidx]*sizeof(C_t), cudaMemcpyDeviceToHost));
            CUDA_CALL(cudaDeviceSynchronize());
        }
        timer->stop_store("Transpose (Packing)");

        for (auto p : comm_order) { 
            // start non-blocking receive for rank p
            MPI_Irecv(&recv_ptr[output_start_z[p]], 1, MPI_PENCILS[p], p, p, comm, &recv_req[p]);    
            
            if (p == comm_order[0])
                timer->stop_store("Transpose (First Send)");

            MPI_Isend(&send_ptr[output_sizes_z[pidx]*output_size_y*input_start_x[p]], 
                sizeof(C_t)*output_sizes_z[pidx]*output_size_y*input_sizes_x[p], MPI_BYTE, p, pidx, comm, &send_req[p]);
        }
    }
}

template<typename T>
void MPIcuFFT_Slab_Z_Then_YX<T>::Peer2Peer_Communication(void *complex_, bool forward) {
    using C_t = typename cuFFT<T>::C_t;
    C_t *recv_ptr, *temp_ptr;
    C_t *complex = cuFFT<T>::complex(complex_);
    temp_ptr = cuFFT<T>::complex(mem_d[0]);

    if (forward) {
        if (cuda_aware)
            recv_ptr = cuFFT<T>::complex(mem_d[0]);
        else 
            recv_ptr = cuFFT<T>::complex(mem_h[0]);

        if (config.send_method == Sync) 
            this->Peer2Peer_Sync(complex_, (void *)recv_ptr);
        else if (config.send_method == Streams)
            this->Peer2Peer_Streams(complex_, (void *)recv_ptr);
        else 
            this->Peer2Peer_MPIType(complex_, (void *)recv_ptr);

        timer->stop_store("Transpose (Start Local Transpose)");
        { 
            // transpose local block
            size_t oslice = output_sizes_z[pidx]*output_size_y*input_start_x[pidx];

            CUDA_CALL(cudaMemcpy2DAsync(&temp_ptr[oslice], sizeof(C_t)*output_sizes_z[pidx],
                &complex[output_start_z[pidx]], sizeof(C_t)*output_size_z, 
                sizeof(C_t)*output_sizes_z[pidx], output_size_y*input_sizes_x[pidx],
                cudaMemcpyDeviceToDevice, streams[pidx]));
        }
        timer->stop_store("Transpose (Start Receive)");
        if (!cuda_aware) { // copy received blocks to device
            int p,count=0;
            do {
                MPI_Waitany(pcnt, recv_req.data(), &p, MPI_STATUSES_IGNORE);
                if (p == MPI_UNDEFINED) 
                    break;
                if (count == 0)
                    timer->stop_store("Transpose (First Receive)");
                if (count == pcnt-2)
                    timer->stop_store("Transpose (Finished Receive)");
                count++;

                size_t oslice = output_sizes_z[pidx]*output_size_y*input_start_x[p];   

                CUDA_CALL(cudaMemcpyAsync(&temp_ptr[oslice], &recv_ptr[oslice],
                    output_sizes_z[pidx]*output_size_y*input_sizes_x[p]*sizeof(C_t), 
                    cudaMemcpyHostToDevice, streams[p]));
            } while(p != MPI_UNDEFINED);
        } else { // just wait for all receives
            MPI_Waitall(pcnt, recv_req.data(), MPI_STATUSES_IGNORE);
            timer->stop_store("Transpose (First Receive)");
            timer->stop_store("Transpose (Finished Receive)");
        }          
        CUDA_CALL(cudaDeviceSynchronize());
        timer->stop_store("Transpose (Unpacking)");
    } else {
        C_t *send_ptr, *recv_ptr;
        C_t *temp_ptr = cuFFT<T>::complex(mem_d[0]);
        C_t *copy_ptr = complex;
        if (cuda_aware) {
            send_ptr = cuFFT<T>::complex(mem_d[0]);
        } else {
            recv_ptr = cuFFT<T>::complex(mem_h[0]);
            send_ptr = cuFFT<T>::complex(mem_h[1]);
        }

        if (config.send_method == MPI_Type) {
            if (cuda_aware) 
                recv_ptr = copy_ptr;

            this->Peer2Peer_MPIType(complex_, (void *)recv_ptr, false);

            // local transpose
            timer->stop_store("Transpose (Start Local Transpose)");
            {
                CUDA_CALL(cudaMemcpy2DAsync(&recv_ptr[output_start_z[pidx]], sizeof(C_t)*output_size_z,
                    &temp_ptr[output_sizes_z[pidx]*output_size_y*input_start_x[pidx]], sizeof(C_t)*output_sizes_z[pidx], 
                    sizeof(C_t)*output_sizes_z[pidx], output_size_y*input_sizes_x[pidx],
                    cuda_aware?cudaMemcpyDeviceToDevice:cudaMemcpyDeviceToHost, streams[pidx]));
            }

            timer->stop_store("Transpose (Start Receive)");      

            MPI_Waitall(pcnt, recv_req.data(), MPI_STATUSES_IGNORE);
            CUDA_CALL(cudaDeviceSynchronize());
            timer->stop_store("Transpose (First Receive)");
            timer->stop_store("Transpose (Finished Receive)");

            if (!cuda_aware) {
                CUDA_CALL(cudaMemcpyAsync(copy_ptr, recv_ptr, sizeof(C_t)*input_sizes_x[pidx]*input_size_y*output_size_z, cudaMemcpyHostToDevice));
                CUDA_CALL(cudaDeviceSynchronize());
            }
            timer->stop_store("Transpose (Unpacking)");
        } else {
            if (cuda_aware)
                recv_ptr = cuFFT<T>::complex(mem_d[1]);

            if (config.send_method == Sync)
                this->Peer2Peer_Sync(complex_, (void *)recv_ptr, false);
            else if (config.send_method == Streams)
                this->Peer2Peer_Streams(complex_, (void *)recv_ptr, false);

            // transpose local block
            timer->stop_store("Transpose (Start Local Transpose)");
            { 
                CUDA_CALL(cudaMemcpy2DAsync(&copy_ptr[output_start_z[pidx]], sizeof(C_t)*output_size_z,
                    &temp_ptr[output_sizes_z[pidx]*output_size_y*input_start_x[pidx]], sizeof(C_t)*output_sizes_z[pidx], 
                    sizeof(C_t)*output_sizes_z[pidx], output_size_y*input_sizes_x[pidx],
                    cudaMemcpyDeviceToDevice, streams[pidx]));
            }

            timer->stop_store("Transpose (Start Receive)");
            int p, count=0;
            do {
                MPI_Waitany(pcnt, recv_req.data(), &p, MPI_STATUSES_IGNORE);
                if (p == MPI_UNDEFINED) 
                    break;
                if (count == 0)
                    timer->stop_store("Transpose (First Receive)");
                if (count == pcnt-2)
                    timer->stop_store("Transpose (Finished Receive)");
                count++;

                CUDA_CALL(cudaMemcpy2DAsync(&copy_ptr[output_start_z[p]], sizeof(C_t)*output_size_z,
                    &recv_ptr[output_start_z[p]*input_size_y*input_sizes_x[pidx]], sizeof(C_t)*output_sizes_z[p], 
                    sizeof(C_t)*output_sizes_z[p], input_size_y*input_sizes_x[pidx],
                    cuda_aware?cudaMemcpyDeviceToDevice:cudaMemcpyHostToDevice, streams[p]));
            } while (p != MPI_UNDEFINED);
            CUDA_CALL(cudaDeviceSynchronize());
            timer->stop_store("Transpose (Unpacking)");
        }
    }
}

template<typename T>
void MPIcuFFT_Slab_Z_Then_YX<T>::All2All_Sync(void *complex_, bool forward) {
    using C_t = typename cuFFT<T>::C_t;
    C_t *complex = cuFFT<T>::complex(complex_);
    C_t *send_ptr, *recv_ptr, *temp_ptr;
    temp_ptr = cuFFT<T>::complex(mem_d[0]);

    if (forward) {
        if (cuda_aware) {
            recv_ptr = temp_ptr;
            send_ptr = cuFFT<T>::complex(mem_d[1]);
        } else {
            recv_ptr = cuFFT<T>::complex(mem_h[0]);
            send_ptr = cuFFT<T>::complex(mem_h[1]);
        }

        for (int p = 0; p < pcnt; p++) { 
        size_t oslice = output_start_z[p]*input_size_y*input_sizes_x[pidx];

        CUDA_CALL(cudaMemcpy2DAsync(&send_ptr[oslice], sizeof(C_t)*output_sizes_z[p],
                &complex[output_start_z[p]], sizeof(C_t)*output_size_z, 
                sizeof(C_t)*output_sizes_z[p], input_size_y*input_sizes_x[pidx],
                cuda_aware?cudaMemcpyDeviceToDevice:cudaMemcpyDeviceToHost, streams[p]));
        }
        CUDA_CALL(cudaDeviceSynchronize());
        timer->stop_store("Transpose (Packing)");

        timer->stop_store("Transpose (Start All2All)");
        MPI_Alltoallv(send_ptr, sendcounts.data(), sdispls.data(), MPI_BYTE, 
            recv_ptr, recvcounts.data(), rdispls.data(), MPI_BYTE, comm);
        timer->stop_store("Transpose (Finished All2All)");

        if (!cuda_aware) {
            CUDA_CALL(cudaMemcpyAsync(temp_ptr, recv_ptr, output_size_x*output_size_y*output_sizes_z[pidx]*sizeof(C_t), cudaMemcpyHostToDevice));
            CUDA_CALL(cudaDeviceSynchronize());
        }
        timer->stop_store("Transpose (Unpacking)");
    } else {
        C_t *temp_ptr = cuFFT<T>::complex(mem_d[0]);
        C_t *copy_ptr = complex;
        if (cuda_aware) {
            send_ptr = temp_ptr;
            recv_ptr = cuFFT<T>::complex(mem_d[1]);
        } else {
            send_ptr = cuFFT<T>::complex(mem_h[1]);
            recv_ptr = cuFFT<T>::complex(mem_h[0]);
        } 

        if (!cuda_aware) {
            CUDA_CALL(cudaMemcpyAsync(send_ptr, temp_ptr, output_size_x*output_size_y*output_sizes_z[pidx]*sizeof(C_t), cudaMemcpyDeviceToHost));
            CUDA_CALL(cudaDeviceSynchronize());
        }
        timer->stop_store("Transpose (Packing)");

        timer->stop_store("Transpose (Start All2All)");
        MPI_Alltoallv(send_ptr, recvcounts.data(), rdispls.data(), MPI_BYTE, 
            recv_ptr, sendcounts.data(), sdispls.data(), MPI_BYTE, comm);
        timer->stop_store("Transpose (Finished All2All)");

        for (int p = 0; p < pcnt; p++) {
            CUDA_CALL(cudaMemcpy2DAsync(&copy_ptr[output_start_z[p]], sizeof(C_t)*output_size_z,
                &recv_ptr[output_start_z[p]*input_size_y*input_sizes_x[pidx]], sizeof(C_t)*output_sizes_z[p], 
                sizeof(C_t)*output_sizes_z[p], input_size_y*input_sizes_x[pidx],
                cuda_aware?cudaMemcpyDeviceToDevice:cudaMemcpyHostToDevice, streams[p]));
        }
        CUDA_CALL(cudaDeviceSynchronize());
        timer->stop_store("Transpose (Unpacking)");
    }
}

template<typename T>
void MPIcuFFT_Slab_Z_Then_YX<T>::All2All_MPIType(void *complex_, bool forward) {
    using C_t = typename cuFFT<T>::C_t;
    C_t *complex = cuFFT<T>::complex(complex_);
    C_t *send_ptr, *recv_ptr, *temp_ptr;
    temp_ptr = cuFFT<T>::complex(mem_d[0]);

    if (forward) {
        if (cuda_aware) {
            recv_ptr = temp_ptr;
            send_ptr = complex;
        } else {
            recv_ptr = cuFFT<T>::complex(mem_h[0]);
            send_ptr = cuFFT<T>::complex(mem_h[1]);
        }

        if (!cuda_aware) {
            CUDA_CALL(cudaMemcpyAsync(send_ptr, complex, output_size_z*input_size_y*input_sizes_x[pidx]*sizeof(C_t), cudaMemcpyDeviceToHost));
            CUDA_CALL(cudaDeviceSynchronize());
        }
        timer->stop_store("Transpose (Packing)");

        timer->stop_store("Transpose (Start All2All)");
        MPI_Alltoallw(send_ptr, sendcounts.data(), sdispls.data(), MPI_PENCILS.data(), 
                        recv_ptr, recvcounts.data(), rdispls.data(), MPI_RECV.data(), comm);
        timer->stop_store("Transpose (Finished All2All)");

        if (!cuda_aware) {
            CUDA_CALL(cudaMemcpyAsync(temp_ptr, recv_ptr, output_sizes_z[pidx]*output_size_y*output_size_x*sizeof(C_t), cudaMemcpyHostToDevice));
            CUDA_CALL(cudaDeviceSynchronize());
        }
        timer->stop_store("Transpose (Unpacking)");
    } else {
        C_t *copy_ptr = complex;
        if (cuda_aware) {
            recv_ptr = copy_ptr;
            send_ptr = temp_ptr;
        } else {
            recv_ptr = cuFFT<T>::complex(mem_h[0]);
            send_ptr = cuFFT<T>::complex(mem_h[1]);
        }

        if (!cuda_aware) {
            CUDA_CALL(cudaMemcpyAsync(send_ptr, temp_ptr, output_sizes_z[pidx]*output_size_y*output_size_x*sizeof(C_t), cudaMemcpyDeviceToHost));
            CUDA_CALL(cudaDeviceSynchronize());
        }
        timer->stop_store("Transpose (Packing)");

        timer->stop_store("Transpose (Start All2All)");
        MPI_Alltoallw(send_ptr, recvcounts.data(), rdispls.data(), MPI_RECV.data(), 
            recv_ptr, sendcounts.data(), sdispls.data(), MPI_PENCILS.data(), comm);
        timer->stop_store("Transpose (Finished All2All)");

        if (!cuda_aware) {
            CUDA_CALL(cudaMemcpyAsync(copy_ptr, recv_ptr, output_size_z*input_size_y*input_sizes_x[pidx]*sizeof(C_t), cudaMemcpyHostToDevice));
            CUDA_CALL(cudaDeviceSynchronize());
        }
        timer->stop_store("Transpose (Unpacking)");
    }
}

template<typename T>
void MPIcuFFT_Slab_Z_Then_YX<T>::All2All_Communication(void *complex_, bool forward) {
    if (config.send_method == MPI_Type) 
        this->All2All_MPIType(complex_, forward);
    else 
        this->All2All_Sync(complex_, forward);
}

template<typename T>
void MPIcuFFT_Slab_Z_Then_YX<T>::execR2C(void *out, const void *in) {
    if (!initialized) 
        return;

    forward = true;

    using R_t = typename cuFFT<T>::R_t;
    using C_t = typename cuFFT<T>::C_t;
    R_t *real    = cuFFT<T>::real(in);
    C_t *complex = cuFFT<T>::complex(out);
    if (fft3d) {
        CUFFT_CALL(cuFFT<T>::execR2C(planR2C, real, complex));
        CUDA_CALL(cudaDeviceSynchronize());
    } else {
        timer->start();
        CUFFT_CALL(cuFFT<T>::execR2C(planR2C, real, complex));
        CUDA_CALL(cudaDeviceSynchronize());
        timer->stop_store("1D FFT Z-Direction");

        C_t *temp_ptr = cuFFT<T>::complex(mem_d[0]);

        if (config.comm_method == Peer2Peer)
            this->Peer2Peer_Communication((void *)complex);
        else 
            this->All2All_Communication((void *)complex);

        if (config.comm_method == Peer2Peer && config.send_method == MPI_Type && cuda_aware) // avoid overwriting send_buffer
            MPI_Waitall(pcnt, send_req.data(), MPI_STATUSES_IGNORE);

        // compute remaining 1d FFT, for cuda-aware recv and temp buffer are identical
        CUFFT_CALL(cuFFT<T>::execC2C(planC2C, temp_ptr, complex, CUFFT_FORWARD));
        CUDA_CALL(cudaDeviceSynchronize());

        timer->stop_store("2D FFT Y-X-Direction");
        if (config.comm_method == Peer2Peer) {
            if (config.send_method == Streams)
                mpisend_thread.join();
            if (!cuda_aware || config.send_method != MPI_Type)
                MPI_Waitall(pcnt, send_req.data(), MPI_STATUSES_IGNORE);
        }
        timer->stop_store("Run complete");
    }
    if (config.warmup_rounds == 0) 
        timer->gather();
    else 
        config.warmup_rounds--;
}

template<typename T>
void MPIcuFFT_Slab_Z_Then_YX<T>::execC2R(void *out, const void *in) {
    if (!initialized) 
        return;

    // needed for Streams second thread
    forward = false;

    using R_t = typename cuFFT<T>::R_t;
    using C_t = typename cuFFT<T>::C_t;

    C_t *complex = cuFFT<T>::complex(in);
    R_t *real    = cuFFT<T>::real(out);

    if (fft3d) {
        CUFFT_CALL(cuFFT<T>::execC2R(planC2R, complex, real));
        CUDA_CALL(cudaDeviceSynchronize());
    } else {
        C_t *temp_ptr = cuFFT<T>::complex(mem_d[0]);
        C_t *copy_ptr = complex;

        timer->start();
        CUFFT_CALL(cuFFT<T>::execC2C(planC2C, complex, temp_ptr, CUFFT_INVERSE));
        CUDA_CALL(cudaDeviceSynchronize());
        timer->stop_store("2D FFT Y-X-Direction");

        if (config.comm_method == Peer2Peer)
            this->Peer2Peer_Communication((void *)complex, false);
        else 
            this->All2All_Communication((void *)complex, false);

        // compute remaining 1d FFT, for cuda-aware recv and temp buffer are identical
        CUFFT_CALL(cuFFT<T>::execC2R(planC2R, copy_ptr, real));
        CUDA_CALL(cudaDeviceSynchronize());
        timer->stop_store("1D FFT Z-Direction");

        if (config.comm_method == Peer2Peer) {
            if (config.send_method == Streams && !cuda_aware)
                mpisend_thread.join();
            MPI_Waitall(pcnt, send_req.data(), MPI_STATUSES_IGNORE);
        }
        timer->stop_store("Run complete");
    }
    if (config.warmup_rounds == 0) 
        timer->gather();
    else 
        config.warmup_rounds--;
}

template class MPIcuFFT_Slab_Z_Then_YX<float>;
template class MPIcuFFT_Slab_Z_Then_YX<double>;