#!/bin/bash
#SBATCH --job-name=cuda_opencv_noise
#SBATCH --output=logs/cuda_opencv_noise.out
#SBATCH --error=logs/cuda_opencv_noise.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:00
#SBATCH --partition=gpu_a40_ext

# Load modules
module purge
module load gcc/12.4.0
module load nvhpc-hpcx-2.20-cuda12/25.1

# Define environment variables for OpenCV installed in $HOME
export LD_LIBRARY_PATH=$HOME/opencv_install/lib64:$LD_LIBRARY_PATH
export PKG_CONFIG_PATH=$HOME/opencv_install/lib/pkgconfig:$PKG_CONFIG_PATH

# Create necessary directories if they don't exist
mkdir -p logs images results

# Compile the CUDA program
nvcc -o decomposeAndFilter decomposeAndFilter.cu \
    -I$HOME/opencv_install/include/opencv4 \
    -L$HOME/opencv_install/lib64 \
    -lopencv_world

# Execute the CUDA program
./decomposeAndFilter 1 3 \
    images/image_4K_noise50.png \
    images/image_4K_noise75.png \
    images/image_4K_noise90.png \
    images/resized_pexels-christian-heitz_4K.png