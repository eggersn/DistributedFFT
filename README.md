# DistributedFFT
Library for Distributed Fast Fourier Transforms for heterogeneous GPU Systems

## Installation
Before building the project, make sure that *CMakeLists.txt* contains your specific CUDA_ARCHITECTURE.

```
$ mkdir build
$ cd build
$ cmake ..
$ cmake --build .
```
Afterwards, there should be the executables "reference", "slab" and "pencil". The commands

```
$ ./reference -h
$ ./slab -h
$ ./pencil -h
```
display the available options. Additionally, they are listed in the following.

### Reference
```
Usage: mpirun -n X [mpi args] reference [options] 
Options (required):
 --input-dim-x [-nx]:   Defines the size of the input data in x-direction.
 --input-dim-y [-ny]:   Defines the size of the input data in y-direction.
 --input-dim-z [-nz]:   Defines the size of the input data in z-direction.
Options (optional):
 --testcase [-t]:       Specifies which test should be executed.
   Available values are:
         --testcase 0:  Each rank sends its input to rank 0, where the complete 3D FFT is computed.
         --testcase 1:  Compares the bandwidth of different MPI communication methods.
           To select the communcation method, use "--opt" (or "-o"). For this testcase, available methods are:
                 --opt 0: MPI Peer2Peer communcation.
                 --opt 0: MPI All2All communcation.
         --testcase 2:  Compares the bandwidth of different MPI sending methods, where the sender has to perform a cudaMemcpy2D (relevant for slab decomposition).
           For this testcase, the arguments "--partition1" (or "-p1") and "--partition2" (or "-p2") are required!
           To select the sending method, use "--opt" (or "-o"). For this testcase, available methods are:
                 --opt 0: The sending rank performs a cudaMemcpy2D for each receiver, using cudaDeviceSync before MPI_Isend.
                 --opt 1: The sending rank performs a cudaMemcpy2D for each receiver, using cudaStreams and cudaCallHostFunc to notify a second thread to call MPI_Isend. Here, MPI has to support MPI_THREAD_MULTIPLE.
                 --opt 2: The sending rank uses a custom MPI_Datatype to avoid using cudaMemcpy2D. If MPI is not CUDA-aware, the sender still has to perform a cudaMemcpy1D (D->H).
         --testcase 3:  Compares the bandwidth of different MPI sending methods, where the sender and receiver have to perform a cudaMemcpy2D (relevant for pencil decomposition).
           For this testcase, the arguments "--partition1" (or "-p1") and "--partition2" (or "-p2") are required!
           To select the sending method, use "--opt" (or "-o"). For this testcase, available methods are:
                 --opt 0: The sending rank performs a cudaMemcpy2D for each receiver, using cudaDeviceSync before MPI_Isend.
                 --opt 1: The sending rank performs a cudaMemcpy2D for each receiver, using cudaStreams and cudaCallHostFunc to notify a second thread to call MPI_Isend. Here, MPI has to support MPI_THREAD_MULTIPLE.
                 --opt 2: The sending rank uses a custom MPI_Datatype to avoid using cudaMemcpy2D. If MPI is not CUDA-aware, the sender still has to perform a cudaMemcpy1D (D->H).
 --opt [-o]:            Specifies which option should be used (depending on the testcase).
 --partition1 [-p1]:    Specifies the number of partitions in x-direction.
 --partition2 [-p2]:    Specifies the number of partitions in y-direction.
 --iterations [-i]:     Specifies how often the given testcase should be repeated. For testcases 1-3, the bandwidth is computed as the average across the number of iterations. Default value is 1.
 --warmup-rounds [-w]:  This value is added to the number of iterations. For a warmup round, the performance metrics are not stored.
 --cuda_aware [-c]:     If set and available, device memory pointer are used to call MPI routines.
 --double_prec [-d]:    If set, double precision is used.
 --benchmark_dir [-b]:  Sets the prefix for the benchmark director (default is ../benchmarks).

Example: 
"mpirun -n 4 reference -nx 256 -ny 256 -nz 256 -t 2 -o 1 -p1 2 -p2 2 -i 10 -c -b ../new_benchmarks"
Here, four MPI processes are started which execute the testcase 2 using option 1 (see above). The input data is of size 256^3, where both x- and y-direction are partitioned (thus, each rank starts with input size 128x128x256). The bandwidht is computed as the average across 10 iterations while CUDA-aware MPI is used. The results are stored in ../new_benchmarks (relative to the build dir).
```
### Slab
```
Usage: mpirun -n P [mpi args] slab [options] 
Options (required):
 --input-dim-x [-nx]:   Defines the size of the input data in x-direction.
 --input-dim-y [-ny]:   Defines the size of the input data in y-direction.
 --input-dim-z [-nz]:   Defines the size of the input data in z-direction.
Options (optional):
 --sequence [-s]:       Defines the sequence of dimensions in which the FFT is computed. Available selections are "ZY_Then_X" (default), "Z_Then_YX" and "Y_Then_ZX"
 --comm-method [-comm]: Specifies whether to use "Peer2Peer" or "All2All" MPI communication.
 --send-method [-snd]:  There are 3 available selections:
        1. Sync:        This is the default option. Here, we use cudaDeviceSync before calling MPI_Isend for each receiving rank.
        2. Streams:     Uses cudaStreams for cudaMemcpyAsync along with cudaCallHostFunc to notify a second thread to call MPI_Isend. This option requires MPI_THREAD_MULTIPLE.
        3. MPI_Type:    Uses MPI_Datatype to avoid using cudaMemcpy2D. If MPI is not CUDA-aware, the sender still has to perform a cudaMemcpy1D (D->H).
 --testcase [-t]:       Specifies which test should be executed.
   Available selections are:
        --testcase 0:   Each rank generates a random input of size (Nx/P) x Ny x Nz (P specified by mpirun).
        --testcase 1:   Rank 0 generates the global input and distributes the slabs while computing the complete 3D FFT. Afterwards rank 0 compares its local result with the distributed result.
        --testcase 2:   Same as testcase 0 for the inverse FFT.
        --testcase 3:   Compute the forward FFT, afterwards the inverse FFT and compare the result with the input data.
        --testcase 4:   Approximate the laplacian of a periodic function with a forward and an inverse FFT and compare the results to the exact result.
 --opt [-o]:            Specifies which option to use.
   Available selections are:
        --opt 0:        Default selection, where no coordinate transformation is performed.
        --opt 1:        Depending on the selected sequence (via "-s"), the algorithm performs a coordinate transform. In general, this enables the sending rank to avoid a cudaMemcpy2D, while requiring it from the receiving rank.
 --iterations [-i]:     Specifies how often the given testcase should be repeated.
 --warmup-rounds [-w]:  This value is added to the number of iterations. For a warmup round, the performance metrics are not stored.
 --cuda_aware [-c]:     If set and available, device memory pointer are used to call MPI routines.
 --double_prec [-d]:    If set, double precision is used.
 --benchmark_dir [-b]:  Sets the prefix for the benchmark director (default is ../benchmarks).

Example: 
"mpirun -n 4 slab -nx 256 -ny 256 -nz 256 -s Z_Then_YX -snd Streams -o 1 -i 10 -c -b ../new_benchmarks"
Here, four MPI processes are started which execute the default testcase using Z_Then_YX as the sequence along with option 1. A sending rank uses the "Streams" method. CUDA-aware MPI is enabled, the algorithm performs 10 iterations of the testcase, and the benchmark results are saved under ../new_benchmarks (relative to the build dir).
```

### Pencil
```
Usage: mpirun -n P [mpi args] pencil [options] 
Options (required):
 --input-dim-x [-nx]:   Defines the size of the input data in x-direction.
 --input-dim-y [-ny]:   Defines the size of the input data in y-direction.
 --input-dim-z [-nz]:   Defines the size of the input data in z-direction.
 --partition1 [-p1]:    Specifies the number of partitions in x-direction.
 --partition2 [-p2]:    Specifies the number of partitions in y-direction.
Options (optional):
 --comm-method1 [-comm1]: Specifies whether to use "Peer2Peer" or "All2All" MPI communication.
 --send-method1 [-snd1]: There are 3 available selections:
        1. Sync:        This is the default option. Here, we use cudaDeviceSync before calling MPI_Isend for each receiving rank.
        2. Streams:     Uses cudaStreams for cudaMemcpyAsync along with cudaCallHostFunc to notify a second thread to call MPI_Isend. This option requires MPI_THREAD_MULTIPLE.
        3. MPI_Type:    Uses MPI_Datatype to avoid using cudaMemcpy2D. If MPI is not CUDA-aware, the sender still has to perform a cudaMemcpy1D (D->H).
 --comm-method2 [-comm2]: Same as --comm-method1 for the second global redistribution.
 --send-method2 [-snd2]: Same as --send-method1 for the second global redistribution.
 --testcase [-t]:       Specifies which test should be executed.
   Available selections are:
        --testcase 0:   Each rank generates a random input of size (Nx/P1) x (Ny/P1) x Nz. Here, P1*P2 = P must hold.
        --testcase 1:   Rank 0 generates the global input and distributes the pencils while computing the complete 3D FFT. Afterwards rank 0 compares its local result with the distributed result. Here, P1*P2+1 = P must hold.
        --testcase 2:   Same as testcase 0 for the inverse FFT.
        --testcase 3:   Compute the forward FFT, afterwards the inverse FFT and compare the result with the input data.
        --testcase 4:   Approximate the laplacian of a periodic function with a forward and an inverse FFT and compare the results to the exact result.
 --opt [-o]:            Specifies which option to use.
   Available selections are:
        --opt 0:        Default selection, where no coordinate transformation is performed. This option requires multiple plans for the 1D-FFT in y-direction.
        --opt 1:        The algorithm performs a coordinate transform. Starting from the default data alignment [x][y][z] (z continuous), the 1D-FFT in z-direction transforms the coordinate system into [z][x][y]. Analogously, the 1D-FFT in y-direction into [y][z][x] and finally the 1D-FFT in x-direction into [x][y][z] again.
 --fft-dim [-f]:        Specifies the number of dimension computed by the algorithm. Available selections are 1, 2, and 3 (default).
 --iterations [-i]:     Specifies how often the given testcase should be repeated.
 --warmup-rounds [-w]:  This value is added to the number of iterations. For a warmup round, the performance metrics are not stored.
 --cuda_aware [-c]:     If set and available, device memory pointer are used to call MPI routines.
 --double_prec [-d]:    If set, double precision is used.
 --benchmark_dir [-b]:  Sets the prefix for the benchmark director (default is ../benchmarks).

Example: 
"mpirun -n 4 pencil -nx 256 -ny 256 -nz 256 -p1 2 -p2 2 -snd Streams -o 1 -i 10 -c -b ../new_benchmarks"
Here, four MPI processes are started which execute the default testcase using option 1. Each rank start with input size 128x128x256. A sending rank uses the "Streams" method. CUDA-aware MPI is enabled, the algorithm performs 10 iterations of the testcase, and the benchmark results are saved under ../new_benchmarks (relative to the build dir).
```
