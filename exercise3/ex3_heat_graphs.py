#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 28 10:48:42 2025

@author: enricobozzetto
"""
import numpy as np
import matplotlib.pyplot as plt


filename = "/Users/enricobozzetto/Desktop/PoliTO/1year/HPC/assignment/graphs/50000/grid_B_50000.csv"
matrix_size = 1024  

def read_matrices_from_csv(filename, size):
    matrices = []
    current_matrix = []

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line == "":
                if current_matrix:
                    matrices.append(np.array(current_matrix))
                    current_matrix = []
            else:
                values = list(map(float, line.split(',')))
                current_matrix.append(values)


        if current_matrix:
            matrices.append(np.array(current_matrix))

    return matrices

def plot_heatmaps(matrices):
    for i, matrix in enumerate(matrices):
        plt.figure(figsize=(6, 6))
        plt.imshow(matrix, cmap='coolwarm', interpolation='nearest')
        plt.colorbar(label='Temperature')
        plt.title(f"Iteration {i*5000}")
        plt.axis('off')
        plt.tight_layout()            
        plt.savefig(f"heatmap_{i:03d}_B.png")
        print(f"Saved heatmap_{i:03d}_B.png")
        plt.close()


matrici = read_matrices_from_csv(filename, matrix_size)
plot_heatmaps(matrici)
