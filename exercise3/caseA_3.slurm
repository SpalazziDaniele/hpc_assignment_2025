#!/bin/bash
#SBATCH --job-name=logs/caseA_3_job        
#SBATCH --output=logs/caseA_3_job.out      
#SBATCH --error=logs/caseA_3_job.err       
#SBATCH --nodes=1                     
#SBATCH --ntasks=1                  
#SBATCH --cpus-per-task=11             
#SBATCH --time=00:05:00               
#SBATCH --partition=cpu_sapphire        


SRC="ex3_caseA.c"
EXE="caseA_3"

echo "Compilation"
gcc -fopenmp -O2 -o $EXE $SRC

if [ $? -ne 0 ]; then
    echo "Error"
    exit 1
fi

echo "Program execution"
srun ./$EXE

echo "Execution completed"

