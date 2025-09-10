#include <math.h>
#include "matrixutils.h"
#include <time.h>

// Function to compare two matrices
int compareMatrices(double *C, double *Creference, int dim){
    for(int i = 0; i < dim; i++){
        for(int j = 0; j < dim; j++){
            double tol = 1e-6;
            if(fabs(C[i * dim + j] - Creference[i * dim + j]) > tol)
                    return 0;
        }
    } 
    return 1;
}

// Main function to check the results of the matrix multiplication
int main(int argc, char* argv[]){
    double *A, *B, *C, *CReference;

    if(argc < 4){
        printf("No enough parameter given: %s <dimension>\n", argv[0]);
        return 2;
    }

    int dim = atoi(argv[1]);
    if(dim <= 0){
        printf("Invalid matrix dimension.\n");
        return 2;
    }

    int rep = atoi(argv[2]);
    if(rep <= 0){
        printf("Invalid number of repetitions.\n");
        return 2;
    }

    int size = atoi(argv[3]);
    if(size <= 0){
        printf("Invalid number of processes.\n");
        return 2;
    }

    char fullInputAFileName[256];
    snprintf(fullInputAFileName, sizeof(fullInputAFileName), "%s_%d_%d.csv", INPUTFILEA, dim, rep);

    char fullInputBFileName[256];
    snprintf(fullInputBFileName, sizeof(fullInputBFileName), "%s_%d_%d.csv", INPUTFILEB, dim, rep);

    char fullReferenceOutputFileName[256];
    snprintf(fullReferenceOutputFileName, sizeof(fullReferenceOutputFileName), "%s_%d_%d.csv", REFERENCEOUTPUTFILE, dim, rep);

    char fullOutputFileName[256];
    snprintf(fullOutputFileName, sizeof(fullOutputFileName), "%s_%d_%d_%d.csv", OUTPUTFILE, dim, rep, size);

    if(dim != readMatrixFile(&A, fullInputAFileName) || dim != readMatrixFile(&B, fullInputBFileName) || dim != readMatrixFile(&C, fullOutputFileName) || dim != readMatrixFile(&CReference, fullReferenceOutputFileName)){
        printf("Error: Matrices dimensions do not match!\n");
        return 2;
    }    

        // Free allocated memory
    if(compareMatrices(C,CReference, dim)){
        free(A);
        free(B);
        free(C);
        free(CReference);
        return 0;
        //printf("Success: Matrices are equal.\n");
    } else {
        free(A);
        free(B);
        free(C);
        free(CReference);
        return 1;
        //printf("Error: Matrices are not equal!\n");
    }
}

