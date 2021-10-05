#!/bin/bash
#SBATCH -p gpu_8
#SBATCH --exclusive
#SBATCH --gres=gpu:8
#SBATCH --nodes=2
#SBATCH --ntasks=16
#SBATCH --time=12:00:00
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
HOST16="$(echo "$HOSTS" | tr '\n' ',' | sed 's/,$//' | sed 's/,/ /g')"
echo $HOST16
HOST8x0="$(echo "$HOSTS" | head -n 8 | tr '\n' ',' | sed 's/,$//' | sed 's/,/ /g')"
echo $HOST8x0
HOST8x1="$(echo "$HOSTS" | tail -n 8 | tr '\n' ',' | sed 's/,$//' | sed 's/,/ /g')"
echo $HOST8x1

# build
echo "start building"
cd $HOME/DistributedFFT/
rm -rf build 
mkdir build
cd build

cmake ..
cmake --build .
echo "finished building"

sleep 5
cd ..

echo "start python script"

echo "Starting on HOST16"
echo "-----------------------------------------------------------------------------"
echo "Slab 2D->1D default"
python launch.py --jobs bwunicluster/slab/benchmarks_base.json bwunicluster/slab/validation.json --hosts $HOST16 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --global_params "-p 16 -b ../benchmarks/bwunicluster/gpu8/small/forward"
echo "Slab 2D->1D opt1"
python launch.py --jobs bwunicluster/slab/benchmarks_base.json bwunicluster/slab/validation.json --hosts $HOST16 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --global_params "-p 16 -b ../benchmarks/bwunicluster/gpu8/small/forward --opt 1"
echo "Slab 1D->2D default"
python launch.py --jobs bwunicluster/slab/benchmarks_base.json bwunicluster/slab/validation.json --hosts $HOST16 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --global_params "-p 16 -b ../benchmarks/bwunicluster/gpu8/small/forward -s Z_Then_YX"
echo "Slab 1D->2D opt1"
python launch.py --jobs bwunicluster/slab/benchmarks_base.json bwunicluster/slab/validation.json --hosts $HOST16 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --global_params "-p 16 -b ../benchmarks/bwunicluster/gpu8/small/forward -s Z_Then_YX --opt 1"
echo "Slab 2D->1D default (inverse)"
python launch.py --jobs bwunicluster/slab/benchmarks_base.json --hosts $HOST16 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --global_params "-t 2 -p 16 -b ../benchmarks/bwunicluster/gpu8/small/inverse"
echo "Slab 2D->1D opt1 (inverse)"
python launch.py --jobs bwunicluster/slab/benchmarks_base.json --hosts $HOST16 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --global_params "-t 2 -p 16 -b ../benchmarks/bwunicluster/gpu8/small/inverse --opt 1"
echo "Slab 1D->2D default (inverse)"
python launch.py --jobs bwunicluster/slab/benchmarks_base.json --hosts $HOST16 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --global_params "-t 2 -p 16 -b ../benchmarks/bwunicluster/gpu8/small/inverse -s Z_Then_YX"
echo "Slab 1D->2D opt1 (inverse)"
python launch.py --jobs bwunicluster/slab/benchmarks_base.json --hosts $HOST16 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --global_params "-t 2 -p 24 -b ../benchmarks/bwunicluster/gpu8/small/inverse -s Z_Then_YX --opt 1"

echo "Starting on HOST8"
echo "-----------------------------------------------------------------------------"

echo "Slab 2D->1D default"
python launch.py --jobs bwunicluster/slab/benchmarks_base.json bwunicluster/slab/validation.json --hosts $HOST8x0 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --global_params "-p 8 -b ../benchmarks/bwunicluster/gpu8/small/forward" --id 1 &
python launch.py --jobs bwunicluster/slab/benchmarks_base.json --hosts $HOST8x1 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --global_params "-t 2 -p 8 -b ../benchmarks/bwunicluster/gpu8/small/inverse"  --id 2 &
wait
echo "Slab 2D->1D opt1"
python launch.py --jobs bwunicluster/slab/benchmarks_base.json bwunicluster/slab/validation.json --hosts $HOST8x0 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --global_params "-p 8 -b ../benchmarks/bwunicluster/gpu8/small/forward --opt 1" --id 1 &
python launch.py --jobs bwunicluster/slab/benchmarks_base.json --hosts $HOST8x1 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --global_params "-t 2 -p 8 -b ../benchmarks/bwunicluster/gpu8/small/inverse --opt 1" --id 2 &
wait
echo "Slab 1D->2D default"
python launch.py --jobs bwunicluster/slab/benchmarks_base.json bwunicluster/slab/validation.json --hosts $HOST8x0 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --global_params "-p 8 -b ../benchmarks/bwunicluster/gpu8/small/forward -s Z_Then_YX" --id 1 &
python launch.py --jobs bwunicluster/slab/benchmarks_base.json --hosts $HOST8x1 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --global_params "-t 2 -p 8 -b ../benchmarks/bwunicluster/gpu8/small/inverse -s Z_Then_YX" --id 2 &
wait 
echo "Slab 1D->2D opt1"
python launch.py --jobs bwunicluster/slab/benchmarks_base.json bwunicluster/slab/validation.json --hosts $HOST8x0 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --global_params "-p 8 -b ../benchmarks/bwunicluster/gpu8/small/forward -s Z_Then_YX --opt 1" --id 1 &
python launch.py --jobs bwunicluster/slab/benchmarks_base.json --hosts $HOST8x1 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --global_params "-t 2 -p 8 -b ../benchmarks/bwunicluster/gpu8/small/inverse -s Z_Then_YX --opt 1" --id 2 &
wait

echo "all done"