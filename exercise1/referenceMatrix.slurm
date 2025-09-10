#!/bin/bash
#SBATCH --job-name=ref_%j         
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --partition=cpu_sapphire
#SBATCH --output=logs/ref_%j.log
#SBATCH --error=logs/ref_%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=s346877@studenti.polito.it

DIM=$1
REPEATS=$2
TOTAL_TIME=0

for ((i=1; i<=REPEATS; i++)); do
    ./initializeMatrices "$DIM" "$i"
    if [ $? -ne 0 ]; then
        echo "Matrix initialization failed"
        exit 1
    fi

    # Run reference computation and capture stdout
    OUTPUT=$(./calculateReferenceMatrix "$DIM" "$i")
    if [ $? -ne 0 ]; then
        echo "Reference calculation failed"
        exit 1
    fi

    # Extract execution time
    REF_TIME=$(echo "$OUTPUT" | grep "ExecTime" | awk '{print $2}')

    # Accumulate
    TOTAL_TIME=$(echo "$TOTAL_TIME + $REF_TIME" | bc -l)
done

# Compute average
AVG_TIME=$(echo "$TOTAL_TIME / $REPEATS" | bc -l)

# Save result: dimension, repetitions, average_time
echo "DIM,REPEATS,AVG_TIME" > results/reference_${DIM}.csv
echo "$DIM,$REPEATS,$AVG_TIME" >> results/reference_${DIM}.csv
