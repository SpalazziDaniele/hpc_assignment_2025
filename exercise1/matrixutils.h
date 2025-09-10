#ifndef CSV_UTILS_H
#define CSV_UTILS_H

#include <stdio.h>
#include <time.h>
#include <stdlib.h>

#define INPUTFILEA "csv/inputA"
#define INPUTFILEB "csv/inputB"
#define OUTPUTFILE "csv/outputC"
#define REFERENCEOUTPUTFILE "results/referenceOutputC"

// Function declaration
int readMatrixFile(double **matrix, const char *filename);
void saveMatrices(double *matrix, int dim, const char *outputFile);
int readTransposedMatrixFile(double **matrix, const char *filename);

#endif