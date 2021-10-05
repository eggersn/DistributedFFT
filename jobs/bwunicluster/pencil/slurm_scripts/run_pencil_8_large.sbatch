#!/bin/bash
#SBATCH -p gpu_8
#SBATCH --exclusive
#SBATCH --gres=gpu:8
#SBATCH --nodes=8
#SBATCH --ntasks=64
#SBATCH --time=40:00:00
#SBATCH --job-name=gpu8_pencil
#SBATCH --output=gpu8_pencil.%j.out
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
HOST8x7="$(echo "$HOSTS" | tail -n 8 | tr '\n' ',' | sed 's/,$//' | sed 's/,/ /g'"
echo "8x7: $HOST8x7"

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
echo "Partition 4x16"
echo "-----------------------------------------------------------------------------"
echo "Pencil Default"
python launch.py --jobs bwunicluster/pencil/benchmarks_base.json bwunicluster/pencil/validation.json --hosts $HOST64 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-p1 4 -p2 16 -b ../benchmarks/bwunicluster/gpu8/large/forward"
echo "Pencil Opt1"
python launch.py --jobs bwunicluster/pencil/benchmarks_base.json bwunicluster/pencil/validation.json --hosts $HOST64 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-p1 4 -p2 16 -b ../benchmarks/bwunicluster/gpu8/large/forward --opt 1"
echo "Pencil Default Inverse"
python launch.py --jobs bwunicluster/pencil/benchmarks_base.json --hosts $HOST64 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-t 2 -p1 4 -p2 16 -b ../benchmarks/bwunicluster/gpu8/large/inverse"
echo "Pencil Opt1 Inverse"
python launch.py --jobs bwunicluster/pencil/benchmarks_base.json --hosts $HOST64 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-t 2 -p1 4 -p2 16 -b ../benchmarks/bwunicluster/gpu8/large/inverse --opt 1"
echo "Partition 8x8"
echo "-----------------------------------------------------------------------------"
echo "Pencil Default"
python launch.py --jobs bwunicluster/pencil/benchmarks_base.json bwunicluster/pencil/validation.json --hosts $HOST64 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-p1 8 -p2 8 -b ../benchmarks/bwunicluster/gpu8/large/forward"
echo "Pencil Opt1"
python launch.py --jobs bwunicluster/pencil/benchmarks_base.json bwunicluster/pencil/validation.json --hosts $HOST64 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-p1 8 -p2 8 -b ../benchmarks/bwunicluster/gpu8/large/forward --opt 1"
echo "Pencil Default Inverse"
python launch.py --jobs bwunicluster/pencil/benchmarks_base.json --hosts $HOST64 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-t 2 -p1 8 -p2 8 -b ../benchmarks/bwunicluster/gpu8/large/inverse"
echo "Pencil Opt1 Inverse"
python launch.py --jobs bwunicluster/pencil/benchmarks_base.json --hosts $HOST64 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-t 2 -p1 8 -p2 8 -b ../benchmarks/bwunicluster/gpu8/large/inverse --opt 1"
echo "Partition 16x4"
echo "-----------------------------------------------------------------------------"
echo "Pencil Default"
python launch.py --jobs bwunicluster/pencil/benchmarks_base.json bwunicluster/pencil/validation.json --hosts $HOST64 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-p1 16 -p2 4 -b ../benchmarks/bwunicluster/gpu8/large/forward" --mpi_params "--mca btl_openib_warn_default_gid_prefix 0" 
echo "Pencil Opt1"
python launch.py --jobs bwunicluster/pencil/benchmarks_base.json bwunicluster/pencil/validation.json --hosts $HOST64 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-p1 16 -p2 4 -b ../benchmarks/bwunicluster/gpu8/large/forward --opt 1" --mpi_params "--mca btl_openib_warn_default_gid_prefix 0" 
echo "Pencil Default Inverse"
python launch.py --jobs bwunicluster/pencil/benchmarks_base.json --hosts $HOST64 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-t 2 -p1 16 -p2 4 -b ../benchmarks/bwunicluster/gpu8/large/inverse" --mpi_params "--mca btl_openib_warn_default_gid_prefix 0" 
echo "Pencil Opt1 Inverse"
python launch.py --jobs bwunicluster/pencil/benchmarks_base.json --hosts $HOST64 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-t 2 -p1 16 -p2 4 -b ../benchmarks/bwunicluster/gpu8/large/inverse --opt 1" --mpi_params "--mca btl_openib_warn_default_gid_prefix 0" 

echo "Starting on 3xHOST48 / 2xHOST16 / 1xHOST8"
echo "*****************************************************************************"
echo "Partition 3x16 / 4x4"
echo "-----------------------------------------------------------------------------"
echo "Pencil Default"
python launch.py --jobs bwunicluster/pencil/benchmarks_base.json bwunicluster/pencil/validation.json --hosts $HOST48 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-p1 3 -p2 16 -b ../benchmarks/bwunicluster/gpu8/large/forward" --id 1 &
python launch.py --jobs bwunicluster/pencil/benchmarks_base.json bwunicluster/pencil/validation.json --hosts $HOST16 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-p1 4 -p2 4 -b ../benchmarks/bwunicluster/gpu8/large/forward" --id 2 --mpi_params "--mca btl_openib_warn_default_gid_prefix 0"  &
wait
echo "Pencil Opt1"
python launch.py --jobs bwunicluster/pencil/benchmarks_base.json bwunicluster/pencil/validation.json --hosts $HOST48 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-p1 3 -p2 16 -b ../benchmarks/bwunicluster/gpu8/large/forward --opt 1" --id 1 &
python launch.py --jobs bwunicluster/pencil/benchmarks_base.json bwunicluster/pencil/validation.json --hosts $HOST16 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-p1 4 -p2 4 -b ../benchmarks/bwunicluster/gpu8/large/forward --opt 1" --id 2 --mpi_params "--mca btl_openib_warn_default_gid_prefix 0"  &
wait
echo "Pencil Default Inverse"
python launch.py --jobs bwunicluster/pencil/benchmarks_base.json --hosts $HOST48 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-t 2 -p1 3 -p2 16 -b ../benchmarks/bwunicluster/gpu8/large/inverse" --id 1 &
python launch.py --jobs bwunicluster/pencil/benchmarks_base.json --hosts $HOST16 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-t 2 -p1 4 -p2 4 -b ../benchmarks/bwunicluster/gpu8/large/inverse" --id 2 --mpi_params "--mca btl_openib_warn_default_gid_prefix 0"  &
wait
echo "Pencil Opt1 Inverse"
python launch.py --jobs bwunicluster/pencil/benchmarks_base.json --hosts $HOST48 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-t 2 -p1 3 -p2 16 -b ../benchmarks/bwunicluster/gpu8/large/inverse --opt 1" --id 1 &
python launch.py --jobs bwunicluster/pencil/benchmarks_base.json --hosts $HOST16 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-t 2 -p1 4 -p2 4 -b ../benchmarks/bwunicluster/gpu8/large/inverse --opt 1" --id 2 --mpi_params "--mca btl_openib_warn_default_gid_prefix 0"  &
wait
echo "Partition 6x8 / 2x8"
echo "-----------------------------------------------------------------------------"
echo "Pencil Default"
python launch.py --jobs bwunicluster/pencil/benchmarks_base.json bwunicluster/pencil/validation.json --hosts $HOST48 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-p1 6 -p2 8 -b ../benchmarks/bwunicluster/gpu8/large/forward" --id 1 &
python launch.py --jobs bwunicluster/pencil/benchmarks_base.json bwunicluster/pencil/validation.json --hosts $HOST16 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-p1 2 -p2 8 -b ../benchmarks/bwunicluster/gpu8/large/forward" --id 2 &
wait
echo "Pencil Opt1"
python launch.py --jobs bwunicluster/pencil/benchmarks_base.json bwunicluster/pencil/validation.json --hosts $HOST48 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-p1 6 -p2 8 -b ../benchmarks/bwunicluster/gpu8/large/forward --opt 1" --id 1 &
python launch.py --jobs bwunicluster/pencil/benchmarks_base.json bwunicluster/pencil/validation.json --hosts $HOST16 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-p1 2 -p2 8 -b ../benchmarks/bwunicluster/gpu8/large/forward --opt 1" --id 2 &
wait
echo "Pencil Default Inverse"
python launch.py --jobs bwunicluster/pencil/benchmarks_base.json --hosts $HOST48 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-t 2 -p1 6 -p2 8 -b ../benchmarks/bwunicluster/gpu8/large/inverse" --id 1 &
python launch.py --jobs bwunicluster/pencil/benchmarks_base.json --hosts $HOST16 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-t 2 -p1 2 -p2 8 -b ../benchmarks/bwunicluster/gpu8/large/inverse" --id 2 &
wait
echo "Pencil Opt1 Inverse"
python launch.py --jobs bwunicluster/pencil/benchmarks_base.json --hosts $HOST48 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-t 2 -p1 6 -p2 8 -b ../benchmarks/bwunicluster/gpu8/large/inverse --opt 1" --id 1 &
python launch.py --jobs bwunicluster/pencil/benchmarks_base.json --hosts $HOST16 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-t 2 -p1 2 -p2 8 -b ../benchmarks/bwunicluster/gpu8/large/inverse --opt 1" --id 2 &
wait
echo "Partition 12x4 / 2x4"
echo "-----------------------------------------------------------------------------"
echo "Pencil Default"
python launch.py --jobs bwunicluster/pencil/benchmarks_base.json bwunicluster/pencil/validation.json --hosts $HOST48 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-p1 12 -p2 4 -b ../benchmarks/bwunicluster/gpu8/large/forward" --id 1 --mpi_params "--mca btl_openib_warn_default_gid_prefix 0"  &
python launch.py --jobs bwunicluster/pencil/benchmarks_base.json bwunicluster/pencil/validation.json --hosts $HOST8x7 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-p1 2 -p2 4 -b ../benchmarks/bwunicluster/gpu8/large/forward" --id 2 --mpi_params "--mca btl_openib_warn_default_gid_prefix 0" &
wait
echo "Pencil Opt1"
python launch.py --jobs bwunicluster/pencil/benchmarks_base.json bwunicluster/pencil/validation.json --hosts $HOST48 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-p1 12 -p2 4 -b ../benchmarks/bwunicluster/gpu8/large/forward --opt 1" --id 1 --mpi_params "--mca btl_openib_warn_default_gid_prefix 0"  &
python launch.py --jobs bwunicluster/pencil/benchmarks_base.json bwunicluster/pencil/validation.json --hosts $HOST8x7 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-p1 2 -p2 4 -b ../benchmarks/bwunicluster/gpu8/large/forward --opt 1" --id 2 --mpi_params "--mca btl_openib_warn_default_gid_prefix 0"  &
wait
echo "Pencil Default Inverse"
python launch.py --jobs bwunicluster/pencil/benchmarks_base.json --hosts $HOST48 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-t 2 -p1 12 -p2 4 -b ../benchmarks/bwunicluster/gpu8/large/inverse" --id 1 --mpi_params "--mca btl_openib_warn_default_gid_prefix 0" &
python launch.py --jobs bwunicluster/pencil/benchmarks_base.json --hosts $HOST8x7 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-t 2 -p1 2 -p2 4 -b ../benchmarks/bwunicluster/gpu8/large/inverse" --id 2 --mpi_params "--mca btl_openib_warn_default_gid_prefix 0" &
wait
echo "Pencil Opt1 Inverse"
python launch.py --jobs bwunicluster/pencil/benchmarks_base.json --hosts $HOST48 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-t 2 -p1 12 -p2 4 -b ../benchmarks/bwunicluster/gpu8/large/inverse --opt 1" --id 1 --mpi_params "--mca btl_openib_warn_default_gid_prefix 0" &
python launch.py --jobs bwunicluster/pencil/benchmarks_base.json --hosts $HOST8x7 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-t 2 -p1 2 -p2 4 -b ../benchmarks/bwunicluster/gpu8/large/inverse --opt 1" --id 2 --mpi_params "--mca btl_openib_warn_default_gid_prefix 0" &
wait

echo "Starting on HOST32"
echo "*****************************************************************************"
echo "Partition 4x8 / 8x4"
echo "-----------------------------------------------------------------------------"
echo "Pencil Default"
python launch.py --jobs bwunicluster/pencil/benchmarks_base.json bwunicluster/pencil/validation.json --hosts $HOST32x0 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-p1 4 -p2 8 -b ../benchmarks/bwunicluster/gpu8/large/forward" --id 1 & 
python launch.py --jobs bwunicluster/pencil/benchmarks_base.json bwunicluster/pencil/validation.json --hosts $HOST32x1 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-p1 8 -p2 4 -b ../benchmarks/bwunicluster/gpu8/large/forward" --id 2 --mpi_params "--mca btl_openib_warn_default_gid_prefix 0" &
wait
echo "Pencil Opt1"
python launch.py --jobs bwunicluster/pencil/benchmarks_base.json bwunicluster/pencil/validation.json --hosts $HOST32x0 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-p1 4 -p2 8 -b ../benchmarks/bwunicluster/gpu8/large/forward --opt 1" --id 1 &
python launch.py --jobs bwunicluster/pencil/benchmarks_base.json bwunicluster/pencil/validation.json --hosts $HOST32x1 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-p1 8 -p2 4 -b ../benchmarks/bwunicluster/gpu8/large/forward --opt 1" --id 2 --mpi_params "--mca btl_openib_warn_default_gid_prefix 0" &
wait
echo "Pencil Default Inverse"
python launch.py --jobs bwunicluster/pencil/benchmarks_base.json --hosts $HOST32x0 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-t 2 -p1 4 -p2 8 -b ../benchmarks/bwunicluster/gpu8/large/inverse" --id 1 &
python launch.py --jobs bwunicluster/pencil/benchmarks_base.json --hosts $HOST32x1 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-t 2 -p1 8 -p2 4 -b ../benchmarks/bwunicluster/gpu8/large/inverse" --id 2 --mpi_params "--mca btl_openib_warn_default_gid_prefix 0" &
wait
echo "Pencil Opt1 Inverse"
python launch.py --jobs bwunicluster/pencil/benchmarks_base.json --hosts $HOST32x0 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-t 2 -p1 4 -p2 8 -b ../benchmarks/bwunicluster/gpu8/large/inverse --opt 1" --id 1 &
python launch.py --jobs bwunicluster/pencil/benchmarks_base.json --hosts $HOST32x1 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-t 2 -p1 8 -p2 4 -b ../benchmarks/bwunicluster/gpu8/large/inverse --opt 1" --id 2 --mpi_params "--mca btl_openib_warn_default_gid_prefix 0" &
wait

echo "all done"