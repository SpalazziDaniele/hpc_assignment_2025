#!/bin/bash
#SBATCH --job-name=systolic_%j    
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --partition=cpu_sapphire
#SBATCH --output=logs/systolic_%j.log
#SBATCH --error=logs/systolic_%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=s346877@studenti.polito.it

DIM=$1
PROCS=$2
REPEATS=$3

module purge
module load openmpi/4.1.8_gcc11
for ((REP=1; REP<=REPEATS; REP++)); do
    # Run MPI program and capture stdout
    OUTPUT=$(mpirun --map-by node -np "$PROCS" ./systolicMatricesMultiplication "$DIM" "$REP")

    # Extract standardized execution time
    SYS_TIME=$(echo "$OUTPUT" | grep "ExecTime" | awk '{print $2}')

    # Check correctness
    ./checkResults "$DIM" "$REP" "$PROCS"
    RET=$?
    STATUS=0
    if [ $RET -eq 1 ]; then STATUS=1; fi   # Not equal
    if [ $RET -eq 2 ]; then STATUS=2; fi   # Check error

    # Save metrics
    echo "$DIM,$PROCS,$REP,$SYS_TIME,$STATUS" >> results/multi_nodes_systolic_${DIM}_${PROCS}_${REP}.csv
done
