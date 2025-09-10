#include <math.h>
#include "matrixutils.h"
#include <time.h>

// Function to evaluate the matrix multiplication
void multiplyMatrices(double *A, double *B, double *C, int dim){
    for(int i = 0; i < dim; i++){
        for(int j = 0; j < dim; j++){
            C[i * dim + j] = 0.0;
            for(int k = 0; k < dim; k++)
                C[i * dim + j] += A[i * dim + k] * B[k * dim + j]; 
        }
    }
}

// Main function to calculate the reference matrix
int main(int argc, char* argv[]){
    double *A, *B, *CReference;

    if(argc < 3){
        printf("No enough parameter given: %s <dimension>\n", argv[0]);
        return 1;
    }

    int dim = atoi(argv[1]);
    if(dim <= 0){
        printf("Invalid matrix dimension.\n");
        return 1;
    }

    int rep = atoi(argv[2]);
    if(rep <= 0){
        printf("Invalid number of repetitions.\n");
        return 1;
    }

    char fullInputAFileName[256];
    snprintf(fullInputAFileName, sizeof(fullInputAFileName), "%s_%d_%d.csv", INPUTFILEA, dim, rep);

    char fullInputBFileName[256];
    snprintf(fullInputBFileName, sizeof(fullInputBFileName), "%s_%d_%d.csv", INPUTFILEB, dim, rep);

    char fullOutputFileName[256];
    snprintf(fullOutputFileName, sizeof(fullOutputFileName), "%s_%d_%d.csv", REFERENCEOUTPUTFILE, dim, rep);

    if(dim != readMatrixFile(&A, fullInputAFileName) || dim != readMatrixFile(&B, fullInputBFileName)){
        printf("Error: Matrices dimensions do not match!\n");
        return 2;
    }

    CReference = malloc (dim * dim * sizeof(double));
    if(CReference == NULL){
        printf("Memory allocation failed!\n");
        return 2;
    }

    clock_t start, end;

    start = clock();
    multiplyMatrices(A,B,CReference, dim);
    end = clock();


    saveMatrices(CReference, dim, fullOutputFileName);

    free(A);
    free(B);
    free(CReference);
    double cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("ExecTime %f\n", cpu_time_used);
    return 0;
}