#!/bin/bash
#SBATCH -p gpu_8
#SBATCH --exclusive
#SBATCH --gres=gpu:8
#SBATCH --ntasks=64
#SBATCH --nodes=8
#SBATCH --time=10:00:00
#SBATCH --job-name=gpu8_slab
#SBATCH --output=gpu8_slab.%j.out
#SBATCH --account=st

# load modules
module load compiler/gnu/8.3.1
module load devel/cuda/11.0
module load devel/cmake/3.18
module load mpi/openmpi/4.1
echo "Modules loaded"

# determine hosts
HOSTS="$(mpirun hostname | sort -n | sed -r 's/\.localdomain//')"
echo "$HOSTS"
HOST64="$(echo "$HOSTS" | tr '\n' ',' | sed 's/,$//' | sed 's/,/ /g')"
echo "64: $HOST64"

# build
echo "start building"
cd $HOME/DistributedFFT/
rm -rf build_gpu8
mkdir build_gpu8
cd build_gpu8

cmake ..
cmake --build .
echo "finished building"

sleep 5
cd ..

echo "start python script"
# start python script
echo "Starting on HOST64"
echo "-----------------------------------------------------------------------------"
echo "Slab 2D->1D default"
python launch.py --jobs bwunicluster/slab/benchmarks_base.json bwunicluster/slab/validation.json --hosts $HOST64 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-p 64 -b ../benchmarks/bwunicluster/gpu8/large/forward "
echo "Slab 2D->1D opt1"
python launch.py --jobs bwunicluster/slab/benchmarks_base.json bwunicluster/slab/validation.json --hosts $HOST64 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-p 64 -b ../benchmarks/bwunicluster/gpu8/large/forward --opt 1"
echo "Slab 1D->2D default"
python launch.py --jobs bwunicluster/slab/benchmarks_base.json bwunicluster/slab/validation.json --hosts $HOST64 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-p 64 -b ../benchmarks/bwunicluster/gpu8/large/forward -s Z_Then_YX"
echo "Slab 1D->2D opt1"
python launch.py --jobs bwunicluster/slab/benchmarks_base.json bwunicluster/slab/validation.json --hosts $HOST64 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-p 64 -b ../benchmarks/bwunicluster/gpu8/large/forward -s Z_Then_YX --opt 1"
echo "Slab 2D->1D default (inverse)"
python launch.py --jobs bwunicluster/slab/benchmarks_base.json --hosts $HOST64 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-t 2 -p 64 -b ../benchmarks/bwunicluster/gpu8/large/inverse"
echo "Slab 2D->1D opt1 (inverse)"
python launch.py --jobs bwunicluster/slab/benchmarks_base.json --hosts $HOST64 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-t 2 -p 64 -b ../benchmarks/bwunicluster/gpu8/large/inverse --opt 1"
echo "Slab 1D->2D default (inverse)"
python launch.py --jobs bwunicluster/slab/benchmarks_base.json --hosts $HOST64 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-t 2 -p 64 -b ../benchmarks/bwunicluster/gpu8/large/inverse -s Z_Then_YX"
echo "Slab 1D->2D opt1 (inverse)"
python launch.py --jobs bwunicluster/slab/benchmarks_base.json --hosts $HOST64 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-t 2 -p 64 -b ../benchmarks/bwunicluster/gpu8/large/inverse -s Z_Then_YX --opt 1"

