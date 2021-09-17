#!/bin/bash
#SBATCH -p gpu_8
#SBATCH --exclusive
#SBATCH --gres=gpu:8
#SBATCH --nodes=6
#SBATCH --ntasks=48
#SBATCH --time=30:00:00
#SBATCH --job-name=gpu8_pencil_med_cuda
#SBATCH --output=gpu8_pencil_med_cuda.%j.out
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
cd /home/st/st_us-051200/st_st160727/DistributedFFT/
rm -rf build_gpu8
mkdir build_gpu8
cd build_gpu8

cmake ..
cmake --build .
echo "finished building"

sleep 5
cd ..

echo "Starting on 3xHOST48"
echo "*****************************************************************************"
echo "Partition 3x16"
echo "-----------------------------------------------------------------------------"
echo "Pencil Default"
python launch.py --jobs bwunicluster/pencil/benchmarks_base0.json bwunicluster/pencil/validation.json --hosts $HOST48 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-c -p1 3 -p2 16 -b ../benchmarks/bwunicluster/gpu8/large/forward"
echo "Pencil Opt1"
python launch.py --jobs bwunicluster/pencil/benchmarks_base0.json bwunicluster/pencil/validation.json --hosts $HOST48 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-c -p1 3 -p2 16 -b ../benchmarks/bwunicluster/gpu8/large/forward --opt 1"
echo "Pencil Default Inverse"
python launch.py --jobs bwunicluster/pencil/benchmarks_base0.json --hosts $HOST48 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-c -t 2 -p1 3 -p2 16 -b ../benchmarks/bwunicluster/gpu8/large/inverse"
echo "Pencil Opt1 Inverse"
python launch.py --jobs bwunicluster/pencil/benchmarks_base0.json --hosts $HOST48 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-c -t 2 -p1 3 -p2 16 -b ../benchmarks/bwunicluster/gpu8/large/inverse --opt 1"
echo "Partition 6x8"
echo "-----------------------------------------------------------------------------"
echo "Pencil Default"
python launch.py --jobs bwunicluster/pencil/benchmarks_base0.json bwunicluster/pencil/validation.json --hosts $HOST48 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-c -p1 6 -p2 8 -b ../benchmarks/bwunicluster/gpu8/large/forward"
echo "Pencil Opt1"
python launch.py --jobs bwunicluster/pencil/benchmarks_base0.json bwunicluster/pencil/validation.json --hosts $HOST48 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-c -p1 6 -p2 8 -b ../benchmarks/bwunicluster/gpu8/large/forward --opt 1"
echo "Pencil Default Inverse"
python launch.py --jobs bwunicluster/pencil/benchmarks_base0.json --hosts $HOST48 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-c -t 2 -p1 6 -p2 8 -b ../benchmarks/bwunicluster/gpu8/large/inverse"
echo "Pencil Opt1 Inverse"
python launch.py --jobs bwunicluster/pencil/benchmarks_base0.json --hosts $HOST48 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-c -t 2 -p1 6 -p2 8 -b ../benchmarks/bwunicluster/gpu8/large/inverse --opt 1"
echo "Partition 12x4"
echo "-----------------------------------------------------------------------------"
echo "Pencil Default"
python launch.py --jobs bwunicluster/pencil/benchmarks_base0.json bwunicluster/pencil/validation.json --hosts $HOST48 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-c -p1 12 -p2 4 -b ../benchmarks/bwunicluster/gpu8/large/forward" --mpi_params "--mca btl_openib_warn_default_gid_prefix 0"
echo "Pencil Opt1"
python launch.py --jobs bwunicluster/pencil/benchmarks_base0.json bwunicluster/pencil/validation.json --hosts $HOST48 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-c -p1 12 -p2 4 -b ../benchmarks/bwunicluster/gpu8/large/forward --opt 1" --mpi_params "--mca btl_openib_warn_default_gid_prefix 0"
echo "Pencil Default Inverse"
python launch.py --jobs bwunicluster/pencil/benchmarks_base0.json --hosts $HOST48 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-c -t 2 -p1 12 -p2 4 -b ../benchmarks/bwunicluster/gpu8/large/inverse" --mpi_params "--mca btl_openib_warn_default_gid_prefix 0"
echo "Pencil Opt1 Inverse"
python launch.py --jobs bwunicluster/pencil/benchmarks_base0.json --hosts $HOST48 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-c -t 2 -p1 12 -p2 4 -b ../benchmarks/bwunicluster/gpu8/large/inverse --opt 1" --mpi_params "--mca btl_openib_warn_default_gid_prefix 0"

echo "Starting on HOST32"
echo "*****************************************************************************"
echo "Partition 4x8 / 4x4"
echo "-----------------------------------------------------------------------------"
echo "Pencil Default"
python launch.py --jobs bwunicluster/pencil/benchmarks_base0.json bwunicluster/pencil/validation.json --hosts $HOST32 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-c -p1 4 -p2 8 -b ../benchmarks/bwunicluster/gpu8/large/forward" --id 1 & 
python launch.py --jobs bwunicluster/pencil/benchmarks_base0.json bwunicluster/pencil/validation.json --hosts $HOST16 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-c -p1 4 -p2 4 -b ../benchmarks/bwunicluster/gpu8/large/forward" --id 2 --mpi_params "--mca btl_openib_warn_default_gid_prefix 0"  &
wait
echo "Pencil Opt1"
python launch.py --jobs bwunicluster/pencil/benchmarks_base0.json bwunicluster/pencil/validation.json --hosts $HOST32 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-c -p1 4 -p2 8 -b ../benchmarks/bwunicluster/gpu8/large/forward --opt 1" --id 1 &
python launch.py --jobs bwunicluster/pencil/benchmarks_base0.json bwunicluster/pencil/validation.json --hosts $HOST16 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-c -p1 4 -p2 4 -b ../benchmarks/bwunicluster/gpu8/large/forward --opt 1" --id 2 --mpi_params "--mca btl_openib_warn_default_gid_prefix 0"  &
wait
echo "Pencil Default Inverse"
python launch.py --jobs bwunicluster/pencil/benchmarks_base0.json --hosts $HOST32 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-c -t 2 -p1 4 -p2 8 -b ../benchmarks/bwunicluster/gpu8/large/inverse" --id 1 &
python launch.py --jobs bwunicluster/pencil/benchmarks_base0.json --hosts $HOST16 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-c -t 2 -p1 4 -p2 4 -b ../benchmarks/bwunicluster/gpu8/large/inverse" --id 2 --mpi_params "--mca btl_openib_warn_default_gid_prefix 0"  &
wait
echo "Pencil Opt1 Inverse"
python launch.py --jobs bwunicluster/pencil/benchmarks_base0.json --hosts $HOST32 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-c -t 2 -p1 4 -p2 8 -b ../benchmarks/bwunicluster/gpu8/large/inverse --opt 1" --id 1 &
python launch.py --jobs bwunicluster/pencil/benchmarks_base0.json --hosts $HOST16 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-c -t 2 -p1 4 -p2 4 -b ../benchmarks/bwunicluster/gpu8/large/inverse --opt 1" --id 2 --mpi_params "--mca btl_openib_warn_default_gid_prefix 0"  &
wait
echo "-----------------------------------------------------------------------------"
echo "Partition 8x4 / 2x8"
echo "-----------------------------------------------------------------------------"
echo "Pencil Default"
python launch.py --jobs bwunicluster/pencil/benchmarks_base0.json bwunicluster/pencil/validation.json --hosts $HOST32 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-c -p1 8 -p2 4 -b ../benchmarks/bwunicluster/gpu8/large/forward" --id 1 --mpi_params "--mca btl_openib_warn_default_gid_prefix 0" &
python launch.py --jobs bwunicluster/pencil/benchmarks_base0.json bwunicluster/pencil/validation.json --hosts $HOST16 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-c -p1 2 -p2 8 -b ../benchmarks/bwunicluster/gpu8/large/forward" --id 2 &
wait
echo "Pencil Opt1"
python launch.py --jobs bwunicluster/pencil/benchmarks_base0.json bwunicluster/pencil/validation.json --hosts $HOST32 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-c -p1 8 -p2 4 -b ../benchmarks/bwunicluster/gpu8/large/forward --opt 1" --id 1 --mpi_params "--mca btl_openib_warn_default_gid_prefix 0" &
python launch.py --jobs bwunicluster/pencil/benchmarks_base0.json bwunicluster/pencil/validation.json --hosts $HOST16 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-c -p1 2 -p2 8 -b ../benchmarks/bwunicluster/gpu8/large/forward --opt 1" --id 2 &
wait
echo "Pencil Default Inverse"
python launch.py --jobs bwunicluster/pencil/benchmarks_base0.json --hosts $HOST32 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-c -t 2 -p1 8 -p2 4 -b ../benchmarks/bwunicluster/gpu8/large/inverse" --id 1 --mpi_params "--mca btl_openib_warn_default_gid_prefix 0" &
python launch.py --jobs bwunicluster/pencil/benchmarks_base0.json --hosts $HOST16 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-c -t 2 -p1 2 -p2 8 -b ../benchmarks/bwunicluster/gpu8/large/inverse" --id 2 &
wait
echo "Pencil Opt1 Inverse"
python launch.py --jobs bwunicluster/pencil/benchmarks_base0.json --hosts $HOST32 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-c -t 2 -p1 8 -p2 4 -b ../benchmarks/bwunicluster/gpu8/large/inverse --opt 1" --id 1 --mpi_params "--mca btl_openib_warn_default_gid_prefix 0" &
python launch.py --jobs bwunicluster/pencil/benchmarks_base0.json --hosts $HOST16 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --build_dir "build_gpu8" --global_params "-c -t 2 -p1 2 -p2 8 -b ../benchmarks/bwunicluster/gpu8/large/inverse --opt 1" --id 2 &
wait

echo "all done"