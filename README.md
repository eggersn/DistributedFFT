# DistributedFFT
Library for Distributed Fast Fourier Transforms for heterogeneous GPU Systems

# Clone Repository
Use the following, to clone the repository **including the full test data**.
```
git clone https://github.com/eggersn/DistributedFFT
```
Alternatively, use the following if you are not interested in the raw test data:
```
git clone -b dev https://github.com/eggersn/DistributedFFT
```

# Building
Before building the project, make sure that *CMakeLists.txt* contains your specific CUDA_ARCHITECTURE.

```
$ mkdir build
$ cd build
$ cmake ..
$ cmake --build .
```
Afterwards, there should be the executables "slab" and "pencil". The help menu can be accessed via:

```
$ ./slab -h
$ ./pencil -h
```
# Defining Testcases

Instead of manually starting each testcase (cf. [Manual Execution](#manual-execution)), multiple testcases can also be described in JSON. The following provides a example for Argon from [here](https://github.com/eggersn/DistributedFFT/blob/master/jobs/argon/slab/benchmarks_base.json). 

Principally, the defined parameters in *global_test_settings* and *tests* simply take available programm parameter, described in (cf. [Manual Execution](#manual-execution)). The parameters defined in *global_test_settings* are used for each test described in *tests*. Furthermore the testcases are repeated for each defined *size*. Additional MPI flags are specified via *additional-flags*, e.g., to specify the hostfile. 

This testcase can later be executed by using either of the following execution methods: [Python Launch Script](#python-launch-script) or [SLURM](#slurm).

```javascript
{
    "size": [128, [128, 128, 256], [128, 256, 256], 256, [256, 256, 512], [256, 512, 512], 512, [512, 512, 1024], [512, 1024, 1024], 1024],
    "additional-flags": "--hostfile ../mpi/hostfile_argon --rankfile ../mpi/rankfile_argon",
    "global_test_settings": {
        "--warmup-rounds": 10,
        "--iterations": 20,
        "--double_prec": true
    },
    "tests": [
        {
            "name": "Slab",
            "-comm": "Peer2Peer",
            "-snd": "Sync",
            "--cuda_aware": false
        }, 
        {
            "name": "Slab",
            "-comm": "Peer2Peer",
            "-snd": "Sync",
            "--cuda_aware": true
        },
        // ...
        {
            "name": "Slab",
            "-comm": "All2All",
            "-snd": "MPI_Type",
            "--cuda_aware": false
        }, 
        {
            "name": "Slab",
            "-comm": "All2All",
            "-snd": "MPI_Type",
            "--cuda_aware": true
        }
    ]
}
```

# Execution

There are three available methods to execute predefined testcases:

- [Python Launch Script](#python-launch-script)
- [SLURM](#slurm)
- [Manual Execution](#manual-execution)

## Python Launch Script
The simplest method is to use the provided launch script:
```
python launch.py
Select a Category:
-----------------------------------
[0] Run Specified Job (job.json)
[1] Run Evaluation Scripts 

Selection: <USER INPUT>
```
By selection option *0*, the user can select predefined jobs from the *jobs* folder (cf. [Defining Testcases](#defining-testcases)). Further programm parameters can be displayed using *python launch.py --help*:
```
usage: launch.py [-h] [--jobs j1 [j1 ...]] [--global_params p] [--mpi_params p]
                 [--hosts h1 [h1 ...]] [--id n] [--gpus n] [--affinity c [c ...]]
                 [--build_dir d]

Launch script for the performance benchmarks.

optional arguments:
  -h, --help            show this help message and exit
  --jobs j1 [j1 ...]    A list of jobs (located in the ./jobs folder), where individual
                        jobs are seperated by spaces. Example "--jobs
                        home/slab/zy_then_x.json home/slab/z_then_yx.json"
  --global_params p     Params passed to slab, pencil or reference MPI call.
  --mpi_params p        Params passed to MPI.
  --hosts h1 [h1 ...]   A list of hostnames seperated by spaces, specifying which hosts
                        to use for MPI execution.
  --id n                Identifier for host- and rankfile in the ./mpi folder. Is
                        required for parallel execution of tests (using this script) to
                        avoid ambiguity.
  --gpus n              Number of GPUs per node.
  --affinity c [c ...]  List of cores for GPU to bind to. The list has to be of length
                        --gpus. Example: "--affinity 0:0-9 1:20-29". Here the first rank
                        is assinged to cores 0-9 on socket 0 for GPU0 and the second
                        rank is assinged to cores 20-29 on socket 1 for GPU1.
  --build_dir d         Path to build directory (default: ./build).

```

## SLURM

Predefined testcases can also be started by using [SLURM](https://slurm.schedmd.com/documentation.html). Exemplary sbatch-scripts can be found in the *jobs* folder. 

We provide an overview of a simple sbatch-script in the following: The script starts by specifying the sbatch parameters and by loading the required modules. Afterwards, the project is rebuild before executing the different testcases. The called testcases (cf. [Define Testcases](#define-testcases)) cover the full spectrum of slab decomposition. For example, the corresponding testcasea of *--jobs argon/slab/benchmarks_base.json* can be found [here](https://github.com/eggersn/DistributedFFT/blob/master/jobs/argon/slab/benchmarks_base.json). Note, that the testcase contains the relevant location of the used hostfile and rankfile. Alternatively, the hostfile can be specified by using *--mpi_params "--hostfile \<location\>"* (cf. [Python Launch Script](#python-launch-script)). The script can be submitted by using *sbatch \<name-of-script\>*.
```bash
#!/bin/bash
#SBATCH -p all
#SBATCH --nodelist=argon-tesla1, argon-tesla2
#SBATCH --exclusive
#SBATCH --ntasks=4
#SBATCH --time=24:00:00
#SBATCH --job-name=slab
#SBATCH --output=slab.%j.out
#SBATCH --account=st

# load modules
module load mpi/u2004/openmpi-4.1.1-cuda
echo "Modules loaded"

# build
echo "start building"
cd /home/eggersn/DistributedFFT/
rm -rf build_argon
mkdir build_argon
cd build_argon

cmake ..
cmake --build .
echo "finished building"

sleep 5
cd ..

echo "start python script"
echo "-----------------------------------------------------------------------------"
echo "Slab 2D->1D default"
python launch.py --jobs argon/slab/benchmarks_base.json argon/slab/validation.json --build_dir "build_argon" --global_params "-p 4 -b ../benchmarks/argon/forward"
echo "Slab 2D->1D opt1"
python launch.py --jobs argon/slab/benchmarks_base.json argon/slab/validation.json --build_dir "build_argon" --global_params "-p 4 -b ../benchmarks/argon/forward --opt 1"
echo "Slab 1D->2D default"
python launch.py --jobs argon/slab/benchmarks_base.json argon/slab/validation.json --build_dir "build_argon" --global_params "-p 4 -b ../benchmarks/argon/forward -s Z_Then_YX"
echo "Slab 1D->2D opt1"
python launch.py --jobs argon/slab/benchmarks_base.json argon/slab/validation.json --build_dir "build_argon" --global_params "-p 4 -b ../benchmarks/argon/forward -s Z_Then_YX --opt 1"
echo "Slab 2D->1D default (inverse)"
python launch.py --jobs argon/slab/benchmarks_base.json --build_dir "build_argon" --global_params "-t 2 -p 4 -b ../benchmarks/argon/inverse"
echo "Slab 2D->1D opt1 (inverse)"
python launch.py --jobs argon/slab/benchmarks_base.json --build_dir "build_argon" --global_params "-t 2 -p 4 -b ../benchmarks/argon/inverse --opt 1"
echo "Slab 1D->2D default (inverse)"
python launch.py --jobs argon/slab/benchmarks_base.json --build_dir "build_argon" --global_params "-t 2 -p 4 -b ../benchmarks/argon/inverse -s Z_Then_YX"
echo "Slab 1D->2D opt1 (inverse)"
python launch.py --jobs argon/slab/benchmarks_base.json --build_dir "build_argon" --global_params "-t 2 -p 4 -b ../benchmarks/argon/inverse -s Z_Then_YX --opt 1"

echo "all done"
```

## Manual Execution
The program can also be directly started by using mpirun. The available commands are summarized in the following:
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

# License
This library is distributed under GNU GENERAL PUBLIC LICENSE Version 3. Please see LICENSE file.
