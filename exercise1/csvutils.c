#include "matrixutils.h"

// Function to read a transposed matrix from a CSV file
int readMatrixFile(double **matrix, const char *filename){
    FILE *file = fopen(filename, "r");
    if(file == NULL){
        printf("Error opening file %s!\n", filename);
        exit(1);
    }

    int rows = 0;
    int oldcolumns = -1;
    int newcolumns = 0;

    int character, lastCharacter = 0;

    while ((character = fgetc(file)) != EOF) {
        if(character == ',') newcolumns++;
        if(character == '\n'){
            newcolumns++; // last value
            rows++;
            if(oldcolumns != -1 && oldcolumns != newcolumns){
                printf("Error in the matrix structure in the CSV file %s!\n", filename);
                printf("Inconsistent number of columns at line %d\n", rows);
                exit(1);
            }
            oldcolumns = newcolumns;
            newcolumns = 0;
        }
        lastCharacter = character;
    }

    if(lastCharacter != '\n'){
        newcolumns++;
        rows++;
        if(oldcolumns == -1){
            oldcolumns = newcolumns; 
        } else if(oldcolumns != newcolumns){
            printf("Error in the matrix structure in the CSV file %s!\n", filename);
            printf("Inconsistent number of columns at last line\n");
            exit(1);
        }
    }

    if(oldcolumns != rows){
        printf("Error in the matrix structure in the CSV file %s!\n", filename);
        printf("The matrix is not square: %d rows and %d columns\n", rows, oldcolumns);
        exit(1);
    }

    rewind(file);

    *matrix = (double *) malloc (rows * rows * sizeof(double));

    if(*matrix == NULL){
        printf("Error allocating memory for matrix in file %s!\n", filename);
        exit(1);
    }

    for(int i = 0; i < rows; i++){
        for(int j = 0; j < oldcolumns; j++){
            if (fscanf(file, "%lf", &(*matrix)[i * rows + j]) != 1) {
                printf("Error reading matrix data from file %s!\n", filename);
                free(*matrix);
                exit(1);
            }
            if (j < oldcolumns - 1) fgetc(file); // consume the comma
        }
    }

    return rows;

}

// Function to read a transposed matrix from a CSV file
int readTransposedMatrixFile(double **matrix, const char *filename){
    FILE *file = fopen(filename, "r");
    if(file == NULL){
        printf("Error opening file %s!\n", filename);
        exit(1);
    }

    int rows = 0;
    int oldcolumns = -1;
    int newcolumns = 0;

    int character, lastCharacter = 0;

    while ((character = fgetc(file)) != EOF) {
        if(character == ',') newcolumns++;
        if(character == '\n'){
            newcolumns++; // last value
            rows++;
            if(oldcolumns != -1 && oldcolumns != newcolumns){
                printf("Error in the matrix structure in the CSV file %s!\n", filename);
                printf("Inconsistent number of columns at line %d\n", rows);
                exit(1);
            }
            oldcolumns = newcolumns;
            newcolumns = 0;
        }
        lastCharacter = character;
    }

    if(lastCharacter != '\n') rows++; // last line

    if(oldcolumns != rows){
        printf("Error in the matrix structure in the CSV file %s!\n", filename);
        printf("The matrix is not square: %d rows and %d columns\n", rows, oldcolumns);
        exit(1);
    }

    rewind(file);

    *matrix = (double *) malloc (rows * rows * sizeof(double));

    if(*matrix == NULL){
        printf("Error allocating memory for matrix in file %s!\n", filename);
        exit(1);
    }

    for(int i = 0; i < rows; i++){
        for(int j = 0; j < oldcolumns; j++){
            if (fscanf(file, "%lf", &(*matrix)[j * rows + i]) != 1) {
                printf("Error reading matrix data from file %s!\n", filename);
 
                free(*matrix);
                exit(1);
            }
            if (j < oldcolumns - 1) fgetc(file); // consume the comma
        }
    }

    return rows;
}


// Function to save in file the matrices
void saveMatrices(double *matrix, int dim, const char *outputFile){
    FILE *file = fopen(outputFile, "w");
    if(file == NULL){
        printf("Error opening file!\n");
        exit(1);
    }

    for(int i = 0; i < dim; i++){
        for(int j = 0; j < dim ; j++){
            fprintf(file, "%lf", matrix[i * dim + j]);
            if(j < dim -1){
                fprintf(file, ",");
            }
        }
        if(i < dim -1){
            fprintf(file, "\n");
        } 
    }

    fclose(file);
}