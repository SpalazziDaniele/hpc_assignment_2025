#!/bin/bash
# This script allow to submit the experiments to slurm for different matrix dimensions and for different number of processes
# on a number of nodes equal to the number of processes. It also compiles the necessary C programs and prepares the workspace.

# Check that at least 3 arguments are passed
if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <NUM_DIMS> <DIM1> [DIM2 ...] <PROC1> [PROC2 ...]"
    echo "At least one dimension and one process required."
    exit 1
fi

# First argument defines how many dimensions to expect
NUM_DIMS=$1
shift 1

# Validate NUM_DIMS
if ! [ "$NUM_DIMS" -gt 0 ] 2>/dev/null; then
    echo "Error: NUM_DIMS must be a positive integer"
    exit 1
fi

# Extract dimensions
dims=("${@:1:$NUM_DIMS}")
shift "$NUM_DIMS"

# Remaining arguments are processes
procs=("$@")

# Validate dimensions
for d in "${dims[@]}"; do
    if ! [ "$d" -gt 0 ] 2>/dev/null; then
        echo "Error: '$d' is not a positive integer (dimension)."
        exit 1
    fi
done

# Validate processes
for p in "${procs[@]}"; do
    if ! [ "$p" -gt 0 ] 2>/dev/null; then
        echo "Error: '$p' is not a positive integer (process count)."
        exit 1
    fi
done

echo "Dimensions: ${dims[*]}"
echo "Processes: ${procs[*]}"

# Prepare the directories for logs, results and input output CSV files
rm -rf logs
mkdir -p logs results csv

# Load necessary modules
module purge
module load openmpi/4.1.8_gcc11

# Compile the programs
echo "Compiling programs..."
mpicc  systolicMatricesMultiplication.c csvutils.c -o systolicMatricesMultiplication
gcc  checkResults.c csvutils.c -o checkResults
echo "Compilation done."

# Function to estimate time based on dimension
get_time_for_dim () {
    local dim=$1
    if   [ "$dim" -le 500 ];  then echo "00:30:00"
    elif [ "$dim" -le 1000 ]; then echo "01:00:00"
    elif [ "$dim" -le 2000 ]; then echo "01:30:00"
    else echo "02:30:00"
    fi
}

# Number of repetitions for each configuration
repeats=10

# Submit jobs for each dimension and processes count
for DIM in "${dims[@]}"; do
        TIME=$(get_time_for_dim "$DIM")
        for P in "${procs[@]}"; do
            sbatch --nodes="$P" --ntasks-per-node=1 --time=$TIME systolicRunMultipleNodes.slurm "$DIM" "$P" "$repeats"
        done
done
