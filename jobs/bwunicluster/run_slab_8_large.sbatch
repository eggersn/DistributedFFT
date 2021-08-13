#!/bin/bash
#SBATCH -p gpu_8
#SBATCH --exclusive
#SBATCH --gres=gpu:8
#SBATCH --ntasks=64
#SBATCH --nodes=8
#SBATCH --time=30:00:00
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
HOST64="$(echo "$HOSTS" | tr '\n' ',' | sed 's/,$//' | sed 's/,/ /g'"
echo "64: $HOST64"
HOST48="$(echo "$HOSTS" | head -n 48 | tr '\n' ',' | sed 's/,$//' | sed 's/,/ /g'"
echo "48: $HOST48"
HOST32x0="$(echo "$HOSTS" | head -n 32 | tr '\n' ',' | sed 's/,$//' | sed 's/,/ /g'"
echo "32x0: $HOST32x0"
HOST32x1="$(echo "$HOSTS" | tail -n 32 | tr '\n' ',' | sed 's/,$//' | sed 's/,/ /g'"
echo "32x1: $HOST32x1"
HOST16="$(echo "$HOSTS" | tail -n 16 | tr '\n' ',' | sed 's/,$//' | sed 's/,/ /g'"
echo "16: $HOST16"
HOST8x0="$(echo "$HOSTS" | head -n 8 | tr '\n' ',' | sed 's/,$//' | sed 's/,/ /g'"
echo "8x0: $HOST8x0"
HOST8x1="$(echo "$HOSTS" | head -n 16 | tail -n 8 | tr '\n' ',' | sed 's/,$//' | sed 's/,/ /g'"
echo "8x1: $HOST8x1"
HOST8x2="$(echo "$HOSTS" | head -n 24 | tail -n 8 | tr '\n' ',' | sed 's/,$//' | sed 's/,/ /g'"
echo "8x2: $HOST8x2"
HOST8x3="$(echo "$HOSTS" | head -n 32 | tail -n 8 | tr '\n' ',' | sed 's/,$//' | sed 's/,/ /g'"
echo "8x3: $HOST8x3"
HOST8x4="$(echo "$HOSTS" | head -n 40 | tail -n 8 | tr '\n' ',' | sed 's/,$//' | sed 's/,/ /g'"
echo "8x4: $HOST8x4"
HOST8x5="$(echo "$HOSTS" | head -n 48 | tail -n 8 | tr '\n' ',' | sed 's/,$//' | sed 's/,/ /g'"
echo "8x5: $HOST8x5"
HOST8x6="$(echo "$HOSTS" | head -n 56 | tail -n 8 | tr '\n' ',' | sed 's/,$//' | sed 's/,/ /g'"
echo "8x6: $HOST8x6"
HOST8x7="$(echo "$HOSTS" | tail -n 8 | tr '\n' ',' | sed 's/,$//' | sed 's/,/ /g'"
echo "8x7: $HOST8x7"

# build
echo "start building"
cd /home/st/st_us-051200/st_st160727/DistributedFFT/
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
python launch.py --jobs bwunicluster/slab/benchmarks_base0.json bwunicluster/slab/validation.json --hosts $HOST64 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-p 64 -b ../benchmarks/forward_gpu8 "
echo "Slab 2D->1D opt1"
python launch.py --jobs bwunicluster/slab/benchmarks_base1.json bwunicluster/slab/validation.json --hosts $HOST64 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-p 64 -b ../benchmarks/forward_gpu8 --opt 1"
echo "Slab 1D->2D default"
python launch.py --jobs bwunicluster/slab/benchmarks_base1.json bwunicluster/slab/validation.json --hosts $HOST64 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-p 64 -b ../benchmarks/forward_gpu8 -s Z_Then_YX"
echo "Slab 1D->2D opt1"
python launch.py --jobs bwunicluster/slab/benchmarks_base1.json bwunicluster/slab/validation.json --hosts $HOST64 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-p 64 -b ../benchmarks/forward_gpu8 -s Z_Then_YX --opt 1"
echo "Slab 2D->1D default (inverse)"
python launch.py --jobs bwunicluster/slab/benchmarks_base1.json bwunicluster/slab/validation.json --hosts $HOST64 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-t 2 -p 64 -b ../benchmarks/inverse_gpu8"
echo "Slab 2D->1D opt1 (inverse)"
python launch.py --jobs bwunicluster/slab/benchmarks_base0.json bwunicluster/slab/validation.json --hosts $HOST64 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-t 2 -p 64 -b ../benchmarks/inverse_gpu8 --opt 1"
echo "Slab 1D->2D default (inverse)"
python launch.py --jobs bwunicluster/slab/benchmarks_base1.json bwunicluster/slab/validation.json --hosts $HOST64 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-t 2 -p 64 -b ../benchmarks/inverse_gpu8 -s Z_Then_YX"
echo "Slab 1D->2D opt1 (inverse)"
python launch.py --jobs bwunicluster/slab/benchmarks_base0.json bwunicluster/slab/validation.json --hosts $HOST64 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-t 2 -p 64 -b ../benchmarks/inverse_gpu8 -s Z_Then_YX --opt 1"

echo "Starting on HOST48 & HOST16"
echo "-----------------------------------------------------------------------------"

echo "Slab 2D->1D default"
python launch.py --jobs bwunicluster/slab/benchmarks_base0.json bwunicluster/slab/validation.json --hosts $HOST48 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-p 48 -b ../benchmarks/forward_gpu8" --id 1 &
python launch.py --jobs bwunicluster/slab/benchmarks_base0.json bwunicluster/slab/validation.json --hosts $HOST16 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-p 16 -b ../benchmarks/forward_gpu8" --id 2 &
wait
echo "Slab 2D->1D opt1"
python launch.py --jobs bwunicluster/slab/benchmarks_base1.json bwunicluster/slab/validation.json --hosts $HOST48 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-p 48 -b ../benchmarks/forward_gpu8 --opt 1" --id 1 &
python launch.py --jobs bwunicluster/slab/benchmarks_base1.json bwunicluster/slab/validation.json --hosts $HOST16 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-p 16 -b ../benchmarks/forward_gpu8 --opt 1" --id 2 &
wait
echo "Slab 1D->2D default"
python launch.py --jobs bwunicluster/slab/benchmarks_base1.json bwunicluster/slab/validation.json --hosts $HOST48 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-p 48 -b ../benchmarks/forward_gpu8 -s Z_Then_YX" --id 1 &
python launch.py --jobs bwunicluster/slab/benchmarks_base1.json bwunicluster/slab/validation.json --hosts $HOST16 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-p 16 -b ../benchmarks/forward_gpu8 -s Z_Then_YX" --id 2 &
wait 
echo "Slab 1D->2D opt1"
python launch.py --jobs bwunicluster/slab/benchmarks_base1.json bwunicluster/slab/validation.json --hosts $HOST48 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-p 48 -b ../benchmarks/forward_gpu8 -s Z_Then_YX --opt 1" --id 1 &
python launch.py --jobs bwunicluster/slab/benchmarks_base1.json bwunicluster/slab/validation.json --hosts $HOST16 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-p 16 -b ../benchmarks/forward_gpu8 -s Z_Then_YX --opt 1" --id 2 &
wait

echo "Slab 2D->1D default (inverse)"
python launch.py --jobs bwunicluster/slab/benchmarks_base1.json bwunicluster/slab/validation.json --hosts $HOST48 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-t 2 -p 48 -b ../benchmarks/inverse_gpu8" --id 1 &
python launch.py --jobs bwunicluster/slab/benchmarks_base1.json bwunicluster/slab/validation.json --hosts $HOST16 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-t 2 -p 16 -b ../benchmarks/inverse_gpu8" --id 2 &
wait
echo "Slab 2D->1D opt1 (inverse)"
python launch.py --jobs bwunicluster/slab/benchmarks_base0.json bwunicluster/slab/validation.json --hosts $HOST48 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-t 2 -p 48 -b ../benchmarks/inverse_gpu8 --opt 1" --id 1 &
python launch.py --jobs bwunicluster/slab/benchmarks_base0.json bwunicluster/slab/validation.json --hosts $HOST16 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-t 2 -p 16 -b ../benchmarks/inverse_gpu8 --opt 1" --id 2 &
wait
echo "Slab 1D->2D default (inverse)"
python launch.py --jobs bwunicluster/slab/benchmarks_base1.json bwunicluster/slab/validation.json --hosts $HOST48 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-t 2 -p 48 -b ../benchmarks/inverse_gpu8 -s Z_Then_YX" --id 1 &
python launch.py --jobs bwunicluster/slab/benchmarks_base1.json bwunicluster/slab/validation.json --hosts $HOST16 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-t 2 -p 16 -b ../benchmarks/inverse_gpu8 -s Z_Then_YX" --id 2 &
wait 
echo "Slab 1D->2D opt1 (inverse)"
python launch.py --jobs bwunicluster/slab/benchmarks_base0.json bwunicluster/slab/validation.json --hosts $HOST48 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-t 2 -p 48 -b ../benchmarks/inverse_gpu8 -s Z_Then_YX --opt 1" --id 1 &
python launch.py --jobs bwunicluster/slab/benchmarks_base0.json bwunicluster/slab/validation.json --hosts $HOST16 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-t 2 -p 16 -b ../benchmarks/inverse_gpu8 -s Z_Then_YX --opt 1" --id 2 &
wait

echo "Starting on HOST32"
echo "-----------------------------------------------------------------------------"

echo "Slab 2D->1D default"
python launch.py --jobs bwunicluster/slab/benchmarks_base0.json bwunicluster/slab/validation.json --hosts $HOST32x0 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-p 32 -b ../benchmarks/forward_gpu8" --id 1 &
python launch.py --jobs bwunicluster/slab/benchmarks_base1.json bwunicluster/slab/validation.json --hosts $HOST32x1 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-t 2 -p 32 -b ../benchmarks/inverse_gpu8" --id 2 &
wait
echo "Slab 2D->1D opt1"
python launch.py --jobs bwunicluster/slab/benchmarks_base1.json bwunicluster/slab/validation.json --hosts $HOST32x0 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-p 32 -b ../benchmarks/forward_gpu8 --opt 1" --id 1 &
python launch.py --jobs bwunicluster/slab/benchmarks_base0.json bwunicluster/slab/validation.json --hosts $HOST32x1 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-t 2 -p 32 -b ../benchmarks/inverse_gpu8 --opt 1" --id 2 &
wait
echo "Slab 1D->2D default"
python launch.py --jobs bwunicluster/slab/benchmarks_base1.json bwunicluster/slab/validation.json --hosts $HOST32x0 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-p 32 -b ../benchmarks/forward_gpu8 -s Z_Then_YX" --id 1 &
python launch.py --jobs bwunicluster/slab/benchmarks_base1.json bwunicluster/slab/validation.json --hosts $HOST32x1 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-t 2 -p 32 -b ../benchmarks/inverse_gpu8 -s Z_Then_YX" --id 2 &
wait 
echo "Slab 1D->2D opt1"
python launch.py --jobs bwunicluster/slab/benchmarks_base1.json bwunicluster/slab/validation.json --hosts $HOST32x0 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-p 32 -b ../benchmarks/forward_gpu8 -s Z_Then_YX --opt 1" --id 1 &
python launch.py --jobs bwunicluster/slab/benchmarks_base0.json bwunicluster/slab/validation.json --hosts $HOST32x1 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-t 2 -p 32 -b ../benchmarks/inverse_gpu8 -s Z_Then_YX --opt 1" --id 2 &
wait

echo "Starting on HOST8"
echo "-----------------------------------------------------------------------------"

python launch.py --jobs bwunicluster/slab/benchmarks_base0.json bwunicluster/slab/validation.json --hosts $HOST8x0 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-p 8 -b ../benchmarks/forward_gpu8" --id 1 &
python launch.py --jobs bwunicluster/slab/benchmarks_base1.json bwunicluster/slab/validation.json --hosts $HOST8x1 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-t 2 -p 8 -b ../benchmarks/inverse_gpu8" --id 2 &
python launch.py --jobs bwunicluster/slab/benchmarks_base1.json bwunicluster/slab/validation.json --hosts $HOST8x2 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-p 8 -b ../benchmarks/forward_gpu8 --opt 1" --id 3 &
python launch.py --jobs bwunicluster/slab/benchmarks_base0.json bwunicluster/slab/validation.json --hosts $HOST8x3 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-t 2 -p 8 -b ../benchmarks/inverse_gpu8 --opt 1" --id 4 &
python launch.py --jobs bwunicluster/slab/benchmarks_base1.json bwunicluster/slab/validation.json --hosts $HOST8x4 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-p 8 -b ../benchmarks/forward_gpu8 -s Z_Then_YX" --id 5 &
python launch.py --jobs bwunicluster/slab/benchmarks_base1.json bwunicluster/slab/validation.json --hosts $HOST8x5 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-t 2 -p 8 -b ../benchmarks/inverse_gpu8 -s Z_Then_YX" --id 6 &
python launch.py --jobs bwunicluster/slab/benchmarks_base1.json bwunicluster/slab/validation.json --hosts $HOST8x6 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-p 8 -b ../benchmarks/forward_gpu8 -s Z_Then_YX --opt 1" --id 7 &
python launch.py --jobs bwunicluster/slab/benchmarks_base0.json bwunicluster/slab/validation.json --hosts $HOST8x7 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-t 2 -p 8 -b ../benchmarks/inverse_gpu8 -s Z_Then_YX --opt 1" --id 8 &
wait