#!/bin/bash
# This script allow to clean up the workspace, compile the C program for the initialization 
# and the reference matrix calculation, and submit the jobs to SLURM for different matrix dimensions
# to prepare the experimental environment.


# Check that at least 3 arguments are passed
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <DIM1> [DIM2 ...]"
    echo "At least one dimension is required."
    exit 1
fi

# Extract dimensions
dims=("${@}")
REPEATS=10

# Validate dimensions
for d in "${dims[@]}"; do
    if ! [ "$d" -gt 0 ] 2>/dev/null; then
        echo "Error: '$d' is not a positive integer (dimension)."
        exit 1
    fi
done

echo "Dimensions: ${dims[*]}"
echo "Repetitions: $REPEATS"

# Clean up and create necessary directories for a new experiment
rm -rf csv results logs
mkdir -p csv results logs

# Compile the C programs
echo "Compiling programs..."
gcc  initializeMatrices.c csvutils.c -o initializeMatrices
gcc  calculateReferenceMatrix.c csvutils.c -o calculateReferenceMatrix
echo "Compilation done."

# Function to determine time based on dimension
get_time_for_dim () {
    local dim=$1
    if   [ "$dim" -le 500 ];  then echo "00:05:00"
    elif [ "$dim" -le 1000 ]; then echo "00:15:00"
    elif [ "$dim" -le 2000 ]; then echo "01:00:00"
    else echo "02:30:00"
    fi
}

# Submit jobs to SLURM for each dimension
for DIM in "${dims[@]}"; do
        TIME=$(get_time_for_dim "$DIM")

        sbatch --parsable --time=$TIME referenceMatrix.slurm "$DIM" "$REPEATS"
done
