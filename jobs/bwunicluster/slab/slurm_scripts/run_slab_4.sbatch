#!/bin/bash
#SBATCH -p gpu_4
#SBATCH --exclusive
#SBATCH --gres=gpu:4
#SBATCH --ntasks=32
#SBATCH --nodes=8
#SBATCH --time=32:00:00
#SBATCH --job-name=gpu4_slab_small
#SBATCH --output=gpu4_slab_small.%j.out
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
HOST32="$(echo "$HOSTS" | tr '\n' ',' | sed 's/,$//' | sed 's/,/ /g')"
echo "32: $HOST32"
HOST24="$(echo "$HOSTS" | head -n 24 | tr '\n' ',' | sed 's/,$//' | sed 's/,/ /g')"
echo "24: $HOST24"
HOST16x0="$(echo "$HOSTS" | head -n 16 | tr '\n' ',' | sed 's/,$//' | sed 's/,/ /g')"
echo "16x0: $HOST16x0"
HOST16x1="$(echo "$HOSTS" | tail -n 16 | tr '\n' ',' | sed 's/,$//' | sed 's/,/ /g')"
echo "16x1: $HOST16x1"
HOST8="$(echo "$HOSTS" | tail -n 8 | tr '\n' ',' | sed 's/,$//' | sed 's/,/ /g')"
echo "8: $HOST8"
HOST4x0="$(echo "$HOSTS" | head -n 4 | tr '\n' ',' | sed 's/,$//' | sed 's/,/ /g')"
echo "4x0: $HOST4x0"
HOST4x1="$(echo "$HOSTS" | head -n 8 | tail -n 4 | tr '\n' ',' | sed 's/,$//' | sed 's/,/ /g')"
echo "4x1: $HOST4x1"
HOST4x2="$(echo "$HOSTS" | head -n 12 | tail -n 4 | tr '\n' ',' | sed 's/,$//' | sed 's/,/ /g')"
echo "4x2: $HOST4x2"
HOST4x3="$(echo "$HOSTS" | head -n 16 | tail -n 4 | tr '\n' ',' | sed 's/,$//' | sed 's/,/ /g')"
echo "4x3: $HOST4x3"
HOST4x4="$(echo "$HOSTS" | head -n 20 | tail -n 4 | tr '\n' ',' | sed 's/,$//' | sed 's/,/ /g')"
echo "4x4: $HOST4x4"
HOST4x5="$(echo "$HOSTS" | head -n 24 | tail -n 4 | tr '\n' ',' | sed 's/,$//' | sed 's/,/ /g')"
echo "4x5: $HOST4x5"
HOST4x6="$(echo "$HOSTS" | head -n 28 | tail -n 4 | tr '\n' ',' | sed 's/,$//' | sed 's/,/ /g')"
echo "4x6: $HOST4x6"
HOST4x7="$(echo "$HOSTS" | head -n 32 | tail -n 4 | tr '\n' ',' | sed 's/,$//' | sed 's/,/ /g')"
echo "4x7: $HOST4x7"

# build
echo "start building"
cd $HOME/DistributedFFT/
rm -rf build_gpu4
mkdir build_gpu4
cd build_gpu4

cmake ..
cmake --build .
echo "finished building"

sleep 5
cd ..

echo "start python script"
# start python script

echo "Starting on HOST32"
echo "-----------------------------------------------------------------------------"
echo "Slab 2D->1D default"
python launch.py --jobs bwunicluster/slab/benchmarks_base.json bwunicluster/slab/validation.json --build_dir "build_gpu4" --hosts $HOST32 --gpus 4 --affinity 0:0-9 0:0-9 1:0-9 1:0-9 --global_params "-p 32 -b ../benchmarks/bwunicluster/gpu4/forward"
echo "Slab 2D->1D opt1"
python launch.py --jobs bwunicluster/slab/benchmarks_base.json bwunicluster/slab/validation.json --build_dir "build_gpu4" --hosts $HOST32 --gpus 4 --affinity 0:0-9 0:0-9 1:0-9 1:0-9 --global_params "-p 32 -b ../benchmarks/bwunicluster/gpu4/forward --opt 1"
echo "Slab 1D->2D default" 
python launch.py --jobs bwunicluster/slab/benchmarks_base.json bwunicluster/slab/validation.json --build_dir "build_gpu4" --hosts $HOST32 --gpus 4 --affinity 0:0-9 0:0-9 1:0-9 1:0-9 --global_params "-p 32 -b ../benchmarks/bwunicluster/gpu4/forward -s Z_Then_YX"
echo "Slab 1D->2D opt1"
python launch.py --jobs bwunicluster/slab/benchmarks_base.json bwunicluster/slab/validation.json --build_dir "build_gpu4" --hosts $HOST32 --gpus 4 --affinity 0:0-9 0:0-9 1:0-9 1:0-9 --global_params "-p 32 -b ../benchmarks/bwunicluster/gpu4/forward -s Z_Then_YX --opt 1"
echo "Slab 2D->1D default (inverse)"
python launch.py --jobs bwunicluster/slab/benchmarks_base.json --build_dir "build_gpu4" --hosts $HOST32 --gpus 4 --affinity 0:0-9 0:0-9 1:0-9 1:0-9 --global_params "-t 2 -p 32 -b ../benchmarks/bwunicluster/gpu4/inverse"
echo "Slab 2D->1D opt1 (inverse)"
python launch.py --jobs bwunicluster/slab/benchmarks_base.json --build_dir "build_gpu4" --hosts $HOST32 --gpus 4 --affinity 0:0-9 0:0-9 1:0-9 1:0-9 --global_params "-t 2 -p 32 -b ../benchmarks/bwunicluster/gpu4/inverse --opt 1"
echo "Slab 1D->2D default (inverse)"
python launch.py --jobs bwunicluster/slab/benchmarks_base.json --build_dir "build_gpu4" --hosts $HOST32 --gpus 4 --affinity 0:0-9 0:0-9 1:0-9 1:0-9 --global_params "-t 2 -p 32 -b ../benchmarks/bwunicluster/gpu4/inverse -s Z_Then_YX"
echo "Slab 1D->2D opt1 (inverse)"
python launch.py --jobs bwunicluster/slab/benchmarks_base.json --build_dir "build_gpu4" --hosts $HOST32 --gpus 4 --affinity 0:0-9 0:0-9 1:0-9 1:0-9 --global_params "-t 2 -p 32 -b ../benchmarks/bwunicluster/gpu4/inverse -s Z_Then_YX --opt 1"


echo "Starting on HOST24 / HOST8"
echo "-----------------------------------------------------------------------------"
echo "Slab 2D->1D default"
python launch.py --jobs bwunicluster/slab/benchmarks_base.json bwunicluster/slab/validation.json --build_dir "build_gpu4" --hosts $HOST24 --gpus 4 --affinity 0:0-9 0:0-9 1:0-9 1:0-9 --global_params "-p 24 -b ../benchmarks/bwunicluster/gpu4/forward" --id 1 & 
python launch.py --jobs bwunicluster/slab/benchmarks_base.json bwunicluster/slab/validation.json --build_dir "build_gpu4" --hosts $HOST8 --gpus 4 --affinity 0:0-9 0:0-9 1:0-9 1:0-9 --global_params "-p 8 -b ../benchmarks/bwunicluster/gpu4/forward" --id 2 & 
wait
echo "Slab 2D->1D opt1"
python launch.py --jobs bwunicluster/slab/benchmarks_base.json bwunicluster/slab/validation.json --build_dir "build_gpu4" --hosts $HOST24 --gpus 4 --affinity 0:0-9 0:0-9 1:0-9 1:0-9 --global_params "-p 24 -b ../benchmarks/bwunicluster/gpu4/forward --opt 1" --id 1 & 
python launch.py --jobs bwunicluster/slab/benchmarks_base.json bwunicluster/slab/validation.json --build_dir "build_gpu4" --hosts $HOST8 --gpus 4 --affinity 0:0-9 0:0-9 1:0-9 1:0-9 --global_params "-p 8 -b ../benchmarks/bwunicluster/gpu4/forward --opt 1" --id 2 & 
wait
echo "Slab 1D->2D default" 
python launch.py --jobs bwunicluster/slab/benchmarks_base.json bwunicluster/slab/validation.json --build_dir "build_gpu4" --hosts $HOST24 --gpus 4 --affinity 0:0-9 0:0-9 1:0-9 1:0-9 --global_params "-p 24 -b ../benchmarks/bwunicluster/gpu4/forward -s Z_Then_YX" --id 1 & 
python launch.py --jobs bwunicluster/slab/benchmarks_base.json bwunicluster/slab/validation.json --build_dir "build_gpu4" --hosts $HOST8 --gpus 4 --affinity 0:0-9 0:0-9 1:0-9 1:0-9 --global_params "-p 8 -b ../benchmarks/bwunicluster/gpu4/forward -s Z_Then_YX" --id 2 & 
wait
echo "Slab 1D->2D opt1"
python launch.py --jobs bwunicluster/slab/benchmarks_base.json bwunicluster/slab/validation.json --build_dir "build_gpu4" --hosts $HOST24 --gpus 4 --affinity 0:0-9 0:0-9 1:0-9 1:0-9 --global_params "-p 24 -b ../benchmarks/bwunicluster/gpu4/forward -s Z_Then_YX --opt 1" --id 1 & 
python launch.py --jobs bwunicluster/slab/benchmarks_base.json bwunicluster/slab/validation.json --build_dir "build_gpu4" --hosts $HOST8 --gpus 4 --affinity 0:0-9 0:0-9 1:0-9 1:0-9 --global_params "-p 8 -b ../benchmarks/bwunicluster/gpu4/forward -s Z_Then_YX --opt 1" --id 2 & 
wait
echo "Slab 2D->1D default (inverse)"
python launch.py --jobs bwunicluster/slab/benchmarks_base.json --build_dir "build_gpu4" --hosts $HOST24 --gpus 4 --affinity 0:0-9 0:0-9 1:0-9 1:0-9 --global_params "-t 2 -p 24 -b ../benchmarks/bwunicluster/gpu4/inverse" --id 1 & 
python launch.py --jobs bwunicluster/slab/benchmarks_base.json --build_dir "build_gpu4" --hosts $HOST8 --gpus 4 --affinity 0:0-9 0:0-9 1:0-9 1:0-9 --global_params "-t 2 -p 8 -b ../benchmarks/bwunicluster/gpu4/inverse" --id 2 & 
wait
echo "Slab 2D->1D opt1 (inverse)"
python launch.py --jobs bwunicluster/slab/benchmarks_base.json --build_dir "build_gpu4" --hosts $HOST24 --gpus 4 --affinity 0:0-9 0:0-9 1:0-9 1:0-9 --global_params "-t 2 -p 24 -b ../benchmarks/bwunicluster/gpu4/inverse --opt 1" --id 1 & 
python launch.py --jobs bwunicluster/slab/benchmarks_base.json --build_dir "build_gpu4" --hosts $HOST8 --gpus 4 --affinity 0:0-9 0:0-9 1:0-9 1:0-9 --global_params "-t 2 -p 8 -b ../benchmarks/bwunicluster/gpu4/inverse --opt 1" --id 2 & 
wait
echo "Slab 1D->2D default (inverse)"
python launch.py --jobs bwunicluster/slab/benchmarks_base.json --build_dir "build_gpu4" --hosts $HOST24 --gpus 4 --affinity 0:0-9 0:0-9 1:0-9 1:0-9 --global_params "-t 2 -p 24 -b ../benchmarks/bwunicluster/gpu4/inverse -s Z_Then_YX" --id 1 &
python launch.py --jobs bwunicluster/slab/benchmarks_base.json --build_dir "build_gpu4" --hosts $HOST8 --gpus 4 --affinity 0:0-9 0:0-9 1:0-9 1:0-9 --global_params "-t 2 -p 8 -b ../benchmarks/bwunicluster/gpu4/inverse -s Z_Then_YX" --id 2 &
wait
echo "Slab 1D->2D opt1 (inverse)"
python launch.py --jobs bwunicluster/slab/benchmarks_base.json --build_dir "build_gpu4" --hosts $HOST24 --gpus 4 --affinity 0:0-9 0:0-9 1:0-9 1:0-9 --global_params "-t 2 -p 24 -b ../benchmarks/bwunicluster/gpu4/inverse -s Z_Then_YX --opt 1" --id 1 & 
python launch.py --jobs bwunicluster/slab/benchmarks_base.json --build_dir "build_gpu4" --hosts $HOST8 --gpus 4 --affinity 0:0-9 0:0-9 1:0-9 1:0-9 --global_params "-t 2 -p 8 -b ../benchmarks/bwunicluster/gpu4/inverse -s Z_Then_YX --opt 1" --id 2 & 
wait

echo "Starting on HOST16"
echo "-----------------------------------------------------------------------------"
echo "Slab 2D->1D default / opt1"
python launch.py --jobs bwunicluster/slab/benchmarks_base.json bwunicluster/slab/validation.json --build_dir "build_gpu4" --hosts $HOST16x0 --gpus 4 --affinity 0:0-9 0:0-9 1:0-9 1:0-9 --global_params "-p 16 -b ../benchmarks/bwunicluster/gpu4/forward" --id 1 &
python launch.py --jobs bwunicluster/slab/benchmarks_base.json bwunicluster/slab/validation.json --build_dir "build_gpu4" --hosts $HOST16x1 --gpus 4 --affinity 0:0-9 0:0-9 1:0-9 1:0-9 --global_params "-p 16 -b ../benchmarks/bwunicluster/gpu4/forward --opt 1" --id 2 &
wait
echo "Slab 1D->2D default / opt1" 
python launch.py --jobs bwunicluster/slab/benchmarks_base.json bwunicluster/slab/validation.json --build_dir "build_gpu4" --hosts $HOST16x0 --gpus 4 --affinity 0:0-9 0:0-9 1:0-9 1:0-9 --global_params "-p 16 -b ../benchmarks/bwunicluster/gpu4/forward -s Z_Then_YX" --id 1 &
python launch.py --jobs bwunicluster/slab/benchmarks_base.json bwunicluster/slab/validation.json --build_dir "build_gpu4" --hosts $HOST16x1 --gpus 4 --affinity 0:0-9 0:0-9 1:0-9 1:0-9 --global_params "-p 16 -b ../benchmarks/bwunicluster/gpu4/forward -s Z_Then_YX --opt 1" --id 2 &
wait
echo "Slab 2D->1D default /opt 1 (inverse)"
python launch.py --jobs bwunicluster/slab/benchmarks_base.json --build_dir "build_gpu4" --hosts $HOST16x0 --gpus 4 --affinity 0:0-9 0:0-9 1:0-9 1:0-9 --global_params "-t 2 -p 16 -b ../benchmarks/bwunicluster/gpu4/inverse" --id 1 &
python launch.py --jobs bwunicluster/slab/benchmarks_base.json --build_dir "build_gpu4" --hosts $HOST16x1 --gpus 4 --affinity 0:0-9 0:0-9 1:0-9 1:0-9 --global_params "-t 2 -p 16 -b ../benchmarks/bwunicluster/gpu4/inverse --opt 1" --id 2 &
wait 
echo "Slab 1D->2D default (inverse)"
python launch.py --jobs bwunicluster/slab/benchmarks_base.json --build_dir "build_gpu4" --hosts $HOST16x0 --gpus 4 --affinity 0:0-9 0:0-9 1:0-9 1:0-9 --global_params "-t 2 -p 16 -b ../benchmarks/bwunicluster/gpu4/inverse -s Z_Then_YX" --id 1 &
python launch.py --jobs bwunicluster/slab/benchmarks_base.json --build_dir "build_gpu4" --hosts $HOST16x1 --gpus 4 --affinity 0:0-9 0:0-9 1:0-9 1:0-9 --global_params "-t 2 -p 16 -b ../benchmarks/bwunicluster/gpu4/inverse -s Z_Then_YX --opt 1" --id 2 &
wait

echo "Starting on HOST4"
echo "-----------------------------------------------------------------------------"
python launch.py --jobs bwunicluster/slab/benchmarks_base.json bwunicluster/slab/validation.json --build_dir "build_gpu4" --hosts $HOST4x0 --gpus 4 --affinity 0:0-9 0:0-9 1:0-9 1:0-9 --global_params "-p 4 -b ../benchmarks/bwunicluster/gpu4/forward" --id 1 & 
python launch.py --jobs bwunicluster/slab/benchmarks_base.json bwunicluster/slab/validation.json --build_dir "build_gpu4" --hosts $HOST4x1 --gpus 4 --affinity 0:0-9 0:0-9 1:0-9 1:0-9 --global_params "-p 4 -b ../benchmarks/bwunicluster/gpu4/forward --opt 1" --id 2 & 
python launch.py --jobs bwunicluster/slab/benchmarks_base.json bwunicluster/slab/validation.json --build_dir "build_gpu4" --hosts $HOST4x2 --gpus 4 --affinity 0:0-9 0:0-9 1:0-9 1:0-9 --global_params "-p 4 -b ../benchmarks/bwunicluster/gpu4/forward -s Z_Then_YX" --id 3 & 
python launch.py --jobs bwunicluster/slab/benchmarks_base.json bwunicluster/slab/validation.json --build_dir "build_gpu4" --hosts $HOST4x3 --gpus 4 --affinity 0:0-9 0:0-9 1:0-9 1:0-9 --global_params "-p 4 -b ../benchmarks/bwunicluster/gpu4/forward -s Z_Then_YX --opt 1" --id 4 & 
python launch.py --jobs bwunicluster/slab/benchmarks_base.json --build_dir "build_gpu4" --hosts $HOST4x4 --gpus 4 --affinity 0:0-9 0:0-9 1:0-9 1:0-9 --global_params "-t 2 -p 4 -b ../benchmarks/bwunicluster/gpu4/inverse" --id 5 & 
python launch.py --jobs bwunicluster/slab/benchmarks_base.json --build_dir "build_gpu4" --hosts $HOST4x5 --gpus 4 --affinity 0:0-9 0:0-9 1:0-9 1:0-9 --global_params "-t 2 -p 4 -b ../benchmarks/bwunicluster/gpu4/inverse --opt 1" --id 6 & 
python launch.py --jobs bwunicluster/slab/benchmarks_base.json --build_dir "build_gpu4" --hosts $HOST4x6 --gpus 4 --affinity 0:0-9 0:0-9 1:0-9 1:0-9 --global_params "-t 2 -p 4 -b ../benchmarks/bwunicluster/gpu4/inverse -s Z_Then_YX" --id 7 & 
python launch.py --jobs bwunicluster/slab/benchmarks_base.json --build_dir "build_gpu4" --hosts $HOST4x7 --gpus 4 --affinity 0:0-9 0:0-9 1:0-9 1:0-9 --global_params "-t 2 -p 4 -b ../benchmarks/bwunicluster/gpu4/inverse -s Z_Then_YX --opt 1" --id 8 & 
wait
