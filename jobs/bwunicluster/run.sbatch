#!/bin/bash
#SBATCH -p "dev_gpu_4"
#SBATCH --exclusive
#SBATCH --gres=gpu:4
#SBATCH --ntasks 4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --job-name=test
#SBATCH --output=test.%j.out

# load modules
module load compiler/gnu/8.3.1
module load devel/cuda/11.0
module load devel/cmake/3.18
module load mpi/openmpi/4.1

# mpi configuration
export OMPI_MCA_btl_openib_allow_ib=1
export OMPI_MCA_btl_openib_if_include="mlx5_0:1"

# determine hosts
HOSTS=`mpirun -n 4 hostname | sort -n | sed -r "s/\.localdomain//"`

HOST64=`echo $HOST | tr "\n" "," | sed "s/,$//" | sed "s/,/ /g"`

# get script location 
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd $SCRIPT_DIR
cd ../../..

# start python script
python launch.py --jobs bwunicluster/slab/zy_then_x.json --hosts $HOST64 --gpus 4 --affinity 0:0-9 0:0-9 1:0-9 1:0-9  

