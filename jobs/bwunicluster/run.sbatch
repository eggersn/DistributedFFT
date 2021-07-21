#!/bin/sh
#SBATCH -p "dev_gpu_4"
#SBATCH --exclusive
#SBATCH --gres=gpu:4
#SBATCH --ntasks 4
#SBATCH --gpu-bind=verbose,map_gpu:0,1,2,3
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --job-name=test
#SBATCH --output=test.%j.out

module load compiler/gnu/8.3.1
module load devel/cuda/11.0
module load devel/cmake/3.18

export PATH="${PATH}:${HOME}/opt/openmpi-4.1.0/bin"

export OMPI_MCA_btl_openib_allow_ib=1
export OMPI_MCA_btl_openib_if_include="mlx5_0:1"

cd ../..
python launch.py bwunicluster/slab/zy_then_x