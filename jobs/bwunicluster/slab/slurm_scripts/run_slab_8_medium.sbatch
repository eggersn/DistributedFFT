#!/bin/bash
#SBATCH -p gpu_8
#SBATCH --exclusive
#SBATCH --gres=gpu:8
#SBATCH --ntasks=48
#SBATCH --nodes=6
#SBATCH --time=20:00:00
#SBATCH --job-name=gpu8_slab_med
#SBATCH --output=gpu8_slab_med.%j.out
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
HOST48="$(echo "$HOSTS" | tr '\n' ',' | sed 's/,$//' | sed 's/,/ /g')"
echo "48: $HOST48"
HOST32="$(echo "$HOSTS" | head -n 32 | tr '\n' ',' | sed 's/,$//' | sed 's/,/ /g')"
echo "32: $HOST32"
HOST16="$(echo "$HOSTS" | tail -n 16 | tr '\n' ',' | sed 's/,$//' | sed 's/,/ /g')"
echo "16: $HOST16"

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
echo "Starting on HOST48"
echo "-----------------------------------------------------------------------------"
echo "Slab 2D->1D default"
python launch.py --jobs bwunicluster/slab/benchmarks_base.json bwunicluster/slab/validation.json --hosts $HOST48 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-p 48 -b ../benchmarks/bwunicluster/gpu8/large/forward "
echo "Slab 2D->1D opt1"
python launch.py --jobs bwunicluster/slab/benchmarks_base.json bwunicluster/slab/validation.json --hosts $HOST48 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-p 48 -b ../benchmarks/bwunicluster/gpu8/large/forward --opt 1"
echo "Slab 1D->2D default"
python launch.py --jobs bwunicluster/slab/benchmarks_base.json bwunicluster/slab/validation.json --hosts $HOST48 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-p 48 -b ../benchmarks/bwunicluster/gpu8/large/forward -s Z_Then_YX"
echo "Slab 1D->2D opt1"
python launch.py --jobs bwunicluster/slab/benchmarks_base.json bwunicluster/slab/validation.json --hosts $HOST48 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-p 48 -b ../benchmarks/bwunicluster/gpu8/large/forward -s Z_Then_YX --opt 1"
echo "Slab 2D->1D default (inverse)"
python launch.py --jobs bwunicluster/slab/benchmarks_base.json --hosts $HOST48 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-t 2 -p 48 -b ../benchmarks/bwunicluster/gpu8/large/inverse"
echo "Slab 2D->1D opt1 (inverse)"
python launch.py --jobs bwunicluster/slab/benchmarks_base.json --hosts $HOST48 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-t 2 -p 48 -b ../benchmarks/bwunicluster/gpu8/large/inverse --opt 1"
echo "Slab 1D->2D default (inverse)"
python launch.py --jobs bwunicluster/slab/benchmarks_base.json --hosts $HOST48 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-t 2 -p 48 -b ../benchmarks/bwunicluster/gpu8/large/inverse -s Z_Then_YX"
echo "Slab 1D->2D opt1 (inverse)"
python launch.py --jobs bwunicluster/slab/benchmarks_base.json --hosts $HOST48 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-t 2 -p 48 -b ../benchmarks/bwunicluster/gpu8/large/inverse -s Z_Then_YX --opt 1"

echo "Starting on HOST32 & HOST16"
echo "-----------------------------------------------------------------------------"

echo "Slab 2D->1D default"
python launch.py --jobs bwunicluster/slab/benchmarks_base.json bwunicluster/slab/validation.json --hosts $HOST32 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-p 32 -b ../benchmarks/bwunicluster/gpu8/large/forward" --id 1 &
python launch.py --jobs bwunicluster/slab/benchmarks_base.json bwunicluster/slab/validation.json --hosts $HOST16 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-p 16 -b ../benchmarks/bwunicluster/gpu8/large/forward" --id 2 &
wait
echo "Slab 2D->1D opt1"
python launch.py --jobs bwunicluster/slab/benchmarks_base.json bwunicluster/slab/validation.json --hosts $HOST32 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-p 32 -b ../benchmarks/bwunicluster/gpu8/large/forward --opt 1" --id 1 &
python launch.py --jobs bwunicluster/slab/benchmarks_base.json bwunicluster/slab/validation.json --hosts $HOST16 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-p 16 -b ../benchmarks/bwunicluster/gpu8/large/forward --opt 1" --id 2 &
wait
echo "Slab 1D->2D default"
python launch.py --jobs bwunicluster/slab/benchmarks_base.json bwunicluster/slab/validation.json --hosts $HOST32 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-p 32 -b ../benchmarks/bwunicluster/gpu8/large/forward -s Z_Then_YX" --id 1 &
python launch.py --jobs bwunicluster/slab/benchmarks_base.json bwunicluster/slab/validation.json --hosts $HOST16 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-p 16 -b ../benchmarks/bwunicluster/gpu8/large/forward -s Z_Then_YX" --id 2 &
wait 
echo "Slab 1D->2D opt1"
python launch.py --jobs bwunicluster/slab/benchmarks_base.json bwunicluster/slab/validation.json --hosts $HOST32 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-p 32 -b ../benchmarks/bwunicluster/gpu8/large/forward -s Z_Then_YX --opt 1" --id 1 &
python launch.py --jobs bwunicluster/slab/benchmarks_base.json bwunicluster/slab/validation.json --hosts $HOST16 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-p 16 -b ../benchmarks/bwunicluster/gpu8/large/forward -s Z_Then_YX --opt 1" --id 2 &
wait

echo "Slab 2D->1D default (inverse)"
python launch.py --jobs bwunicluster/slab/benchmarks_base.json --hosts $HOST32 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-t 2 -p 32 -b ../benchmarks/bwunicluster/gpu8/large/inverse" --id 1 &
python launch.py --jobs bwunicluster/slab/benchmarks_base.json --hosts $HOST16 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-t 2 -p 16 -b ../benchmarks/bwunicluster/gpu8/large/inverse" --id 2 &
wait
echo "Slab 2D->1D opt1 (inverse)"
python launch.py --jobs bwunicluster/slab/benchmarks_base.json --hosts $HOST32 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-t 2 -p 32 -b ../benchmarks/bwunicluster/gpu8/large/inverse --opt 1" --id 1 &
python launch.py --jobs bwunicluster/slab/benchmarks_base.json --hosts $HOST16 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-t 2 -p 16 -b ../benchmarks/bwunicluster/gpu8/large/inverse --opt 1" --id 2 &
wait
echo "Slab 1D->2D default (inverse)"
python launch.py --jobs bwunicluster/slab/benchmarks_base.json --hosts $HOST32 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-t 2 -p 32 -b ../benchmarks/bwunicluster/gpu8/large/inverse -s Z_Then_YX" --id 1 &
python launch.py --jobs bwunicluster/slab/benchmarks_base.json --hosts $HOST16 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-t 2 -p 16 -b ../benchmarks/bwunicluster/gpu8/large/inverse -s Z_Then_YX" --id 2 &
wait 
echo "Slab 1D->2D opt1 (inverse)"
python launch.py --jobs bwunicluster/slab/benchmarks_base.json --hosts $HOST32 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-t 2 -p 32 -b ../benchmarks/bwunicluster/gpu8/large/inverse -s Z_Then_YX --opt 1" --id 1 &
python launch.py --jobs bwunicluster/slab/benchmarks_base.json --hosts $HOST16 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-t 2 -p 16 -b ../benchmarks/bwunicluster/gpu8/large/inverse -s Z_Then_YX --opt 1" --id 2 &
wait