#include "matrixutils.h"


// Function to initialize matrices A, B with random values
void initializeMatrices(double *A, double *B, int dim){
    for(int i = 0; i < dim; i++){
        for(int j = 0; j < dim; j++){
            double r = (double)rand() / RAND_MAX;
            if (r < 0.05) {
                A[i * dim + j] = 0.0;   // 5% chance of being exactly 0
            } else {
                A[i * dim + j] = ((double)rand() / RAND_MAX) * 200.0 - 100.0; // range [-100,100]
            }

            r = (double)rand() / RAND_MAX;
            if (r < 0.05) {
                B[i * dim + j] = 0.0;
            } else {
                B[i * dim + j] = ((double)rand() / RAND_MAX) * 200.0 - 100.0;
            }

        }
    }
}



int main(int argc, char* argv[]){
    
    //printf("This program allow to create two square matrix of the specified dimension and save them in two file inputA.csv and inputB.csv.\n");

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
    srand(time(NULL)+rep+dim);

    char fullInputAFileName[256];
    snprintf(fullInputAFileName, sizeof(fullInputAFileName), "%s_%d_%d.csv", INPUTFILEA, dim, rep);

    char fullInputBFileName[256];
    snprintf(fullInputBFileName, sizeof(fullInputBFileName), "%s_%d_%d.csv", INPUTFILEB, dim, rep);

    double *A, *B;

    A = malloc(dim * dim * sizeof(double));
    B = malloc(dim * dim * sizeof(double));
    if (A == NULL || B == NULL) {
        printf("Memory allocation failed!\n");
        return 1;
    }

    initializeMatrices(A, B, dim);
    saveMatrices(A, dim, fullInputAFileName);
    saveMatrices(B, dim, fullInputBFileName);

    free(A);
    free(B);

    printf("Matrices created and saved successfully.\n");

    return 0;
}