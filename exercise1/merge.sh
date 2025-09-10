#!/bin/bash

# Merge reference CSVs
echo "Merging reference results..."
echo "DIM,REP,TIME" > results/merged_reference.csv
cat results/reference_*.csv 2>/dev/null >> results/merged_reference.csv

# Merge systolic CSVs
echo "Merging systolic results..."
echo "DIM,PROCS,REP,TIME,STATUS" > results/merged_systolic.csv
cat results/systolic_*.csv 2>/dev/null >> results/merged_systolic.csv

#Merge multi-node systolic CSVs
echo "Merging multi-node systolic results..."
echo "DIM,PROCS,REP,TIME,STATUS" > results/merged_multi_nodes_systolic.csv
cat results/multi_nodes_systolic_*.csv 2>/dev/null >> results/merged_multi_nodes_systolic.csv


echo "Merging done. Merged files are in results/merged_reference.csv and results/merged_systolic.csv"
