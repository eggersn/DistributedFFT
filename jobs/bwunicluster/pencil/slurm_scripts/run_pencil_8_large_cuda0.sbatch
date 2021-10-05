#!/bin/bash
#SBATCH -p gpu_8
#SBATCH --exclusive
#SBATCH --gres=gpu:8
#SBATCH --nodes=8
#SBATCH --ntasks=64
#SBATCH --time=06:00:00
#SBATCH --job-name=gpu8_pencil_cuda
#SBATCH --output=gpu8_pencil_cuda.%j.out
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

echo "Starting on HOST64"
echo "*****************************************************************************"
echo "Partition 8x8"
echo "-----------------------------------------------------------------------------"
echo "Pencil Default"
python launch.py --jobs bwunicluster/pencil/benchmarks_base.json bwunicluster/pencil/validation.json --hosts $HOST64 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-c -p1 8 -p2 8 -b ../benchmarks/bwunicluster/gpu8/large/forward"
echo "Pencil Opt1"
python launch.py --jobs bwunicluster/pencil/benchmarks_base.json bwunicluster/pencil/validation.json --hosts $HOST64 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-c -p1 8 -p2 8 -b ../benchmarks/bwunicluster/gpu8/large/forward --opt 1"
echo "Pencil Default Inverse"
python launch.py --jobs bwunicluster/pencil/benchmarks_base.json --hosts $HOST64 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-c -t 2 -p1 8 -p2 8 -b ../benchmarks/bwunicluster/gpu8/large/inverse"
echo "Pencil Opt1 Inverse"
python launch.py --jobs bwunicluster/pencil/benchmarks_base.json --hosts $HOST64 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-c -t 2 -p1 8 -p2 8 -b ../benchmarks/bwunicluster/gpu8/large/inverse --opt 1"


echo "all done"