# DistributedFFT
Library for Distributed Fast Fourier Transforms for heterogeneous GPU Systems

Before building the project, make sure that *CMakeLists.txt* contains your specific CUDA_ARCHITECTURE.
Furthermore, the given tests contain a flag called *ALLOW_CUDA_AWARE*. If set to 1, make sure that you compiled CUDA-aware OpenMPI (see https://www.open-mpi.org/faq/?category=buildcuda)

```
$ mkdir build
$ cd build
$ cmake ..
$ cmake --build .
```

### Example 1: Slab decomposition
Here, the last MPI generates the input, and verifies that the other 3 processes computed the correct result
```
$ mpirun -n 4 slab_random_dist
```

### Example 2: Slab decomposition (oversubscribe)
```
$ mpirun -n 12 --oversubscribe slab_random_dist
```

### Example 3: Pencil decomposition 3D (P1 = 4, P2 = 4)
Ensure that P1*P2+1 = #Num_Of_MPI_Processes (set in e.g. tests/pencil/random_dist_3D)
```
$ mpirun -n 17 --oversubscribe pencil_random_dist3
```

### Example 4: Pencil decomposition 2D (P1 = 4, P2 = 4)
Computes FFT in z- and y-direction: Here, one has to uncomment 
```c
// used for random_dist_2D test
// return;
```
in src/mpicufft_pencil.cpp
```
$ mpirun -n 17 --oversubscribe pencil_random_dist2
```
