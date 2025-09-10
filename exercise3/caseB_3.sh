#!/bin/bash
SRC="caseB_3.c"
EXE="caseB_3"

echo "Compilation"
gcc -fopenmp -O2 -o $EXE $SRC
if [ $? -ne 0 ]; then
    echo "Error"
    exit 1
fi

echo "Program execution"
./$EXE

echo "Execution completed"

