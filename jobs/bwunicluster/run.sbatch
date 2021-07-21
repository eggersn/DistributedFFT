#!/bin/sh
#SBATCH -p "dev_gpu_4"
#SBATCH --exclusive
#SBATCH --gres=gpu:4
#SBATCH --ntasks 4
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=verbose,map_gpu:0*4,1*4,2*4,3*4
#SBATCH --cpus-per-task=2
#SBATCH --time=00:03:00
#SBATCH --job-name=test
#SBATCH --output=test.%j.out

module load compiler/gnu/8.3.1
module load mpi/openmpi/4.1
module load devel/cuda/11.0
module load devel/cmake/3.18

export OMPI_MCA_btl_openib_allow_ib=1
export OMPI_MCA_btl_openib_if_include="mlx5_0:1"

cd DistributedFFT/build
mpirun --report-bindings -n 2 reference --testcase 1 --warmup-rounds 10 --iterations 20 --double_prec -nx 256 -ny 256 -nz 256 -c
mpirun --report-bindings -n 2 reference --testcase 1 --warmup-rounds 10 --iterations 20 --double_prec -nx 256 -ny 256 -nz 256