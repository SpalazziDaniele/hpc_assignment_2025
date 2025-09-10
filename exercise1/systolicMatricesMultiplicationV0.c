#include "matrixutils.h"
#include <mpi.h>
#include <string.h>


int getBaseMatrixJCoord(int matrixOrder, int processOrganizationDims, int remainder, int processCoords){
    int jCoord = 0;
    if(processCoords>=(remainder))
        jCoord = matrixOrder / processOrganizationDims * (processCoords-remainder)+ remainder * (matrixOrder / processOrganizationDims + 1);
    else
        jCoord = (matrixOrder / processOrganizationDims + 1) * processCoords;
    return jCoord;
}

int main(int argc, char** argv){
    // Declarations of the matrices
    double *A = NULL, *BTranspose = NULL, *C = NULL;
    // Declarations of important dimensions
    int rank, size, matrixOrder;
    // Declarations of time dimensions
    double start, end;
    // Initialization of the communicator environment
    MPI_Init(&argc, &argv);
    // Get the rank and size of the communicator
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);


    // Define the 2D grid dimension based on the size of the processes
    int processOrganizationDims[2] = {0, 0};
    MPI_Dims_create(size, 2, processOrganizationDims);

    // Define the communicator for Cartesian organization and initialize the communicator with Cart_create
    MPI_Comm cartComm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, processOrganizationDims, (int[]){0,0}, 1, &cartComm);

    // Get the coordinates of the current process in the Cartesian grid
    int actualprocessCoords[2] = {0, 0};
    MPI_Cart_coords(cartComm, rank, 2, actualprocessCoords);

    // Rank 0 process read the input and initialize the matrices
    if(rank == 0){
        matrixOrder = readMatrixFile(&A, INPUTFILEA);

        // Check that the dimensions match
        if(matrixOrder != readTransposedMatrixFile(&BTranspose, INPUTFILEB)){
            printf("Error: Matrices dimensions do not match!\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
    }

    MPI_Barrier(cartComm);
    start = MPI_Wtime();

    // Broadcast the order of the matrices to all processs
    MPI_Bcast(&matrixOrder, 1, MPI_INT, 0, cartComm);

    // Evaluate the remainder between the matrix order and the dimensions of the processes grid
    int remainder[2] = {
                        matrixOrder % processOrganizationDims[0],
                        matrixOrder % processOrganizationDims[1]
                        };

    // Calculate the local order of the output matrix for different processes
    int localOrder[2] = {
                        matrixOrder / processOrganizationDims[0] + (actualprocessCoords[0] < (remainder[0]) ? 1 : 0),
                        matrixOrder / processOrganizationDims[1] + (actualprocessCoords[1] < (remainder[1]) ? 1 : 0)
                        };

    // Evaluate the coords of the C matrix to save in the local process                    
    int baseMatrixCoords[2] = {
                            getBaseMatrixJCoord(matrixOrder, processOrganizationDims[0], remainder[0], actualprocessCoords[0]), 
                            getBaseMatrixJCoord(matrixOrder, processOrganizationDims[1], remainder[1], actualprocessCoords[1])
                            };
    

    // Rank 0 process send the rows of A to the first column of PE and B columns to the first row of the PE to initialize the systolic evaluation
    if(rank == 0){
        // Send B columns to the first row of PE
        for(int j = 1; j < processOrganizationDims[1]; j++){
            int destCoords[2] = {0,j};
            int destRank;
            MPI_Cart_rank(cartComm, destCoords, &destRank);

            // Evaluate the number of cols that the destination process of the first row need to get
            int numCols = matrixOrder / processOrganizationDims[1] + (j < remainder[1] ? 1 : 0);
            int numElementsToSend = matrixOrder * numCols;

            // Evaluate flattened coordinate of the first column to send
            int baseStartToSend = getBaseMatrixJCoord(matrixOrder, processOrganizationDims[1], remainder[1], j);
            baseStartToSend *= matrixOrder;

            // Send the data
            MPI_Send(&BTranspose[baseStartToSend], numElementsToSend, MPI_DOUBLE, destRank, 0, cartComm);
        }

        // Send A rows to the first column of PE
        for(int i = 1; i < processOrganizationDims[0]; i++){
            int destCoords[2] = {i, 0};
            int destRank;
            MPI_Cart_rank(cartComm, destCoords, &destRank);

            // Evaluate the number of rows that the destination process of the first column need to get
            int numRows = matrixOrder / processOrganizationDims[0] + (i < remainder[0] ? 1 : 0);
            int numElementsToSend = numRows * matrixOrder;

            // Evaluate flattened coordinate of the first row to send
            int baseStartToSend = getBaseMatrixJCoord(matrixOrder, processOrganizationDims[0], remainder[0], i);
            baseStartToSend *= matrixOrder;

            // Send the data
            MPI_Send(&A[baseStartToSend], numElementsToSend, MPI_DOUBLE, destRank, 0, cartComm);
        }
    }

    // Define the local submatrices for the first row and column of the PE
    double *localA = NULL;
    double *localBTranspose = NULL;

    // First column receive the A rows
    if(actualprocessCoords[1] == 0 && rank != 0){
        localA = malloc(localOrder[0] * matrixOrder * sizeof(double));
        if(localA == NULL){
            printf("Memory allocation failed!\n");
            return 1;
        }
        MPI_Recv(localA, localOrder[0] * matrixOrder, MPI_DOUBLE, 0, 0, cartComm, MPI_STATUS_IGNORE);
    }

    // First row receive the B columns
    if(actualprocessCoords[0] == 0 && rank != 0){
        localBTranspose = malloc(localOrder[1] * matrixOrder * sizeof(double));
        if(localBTranspose == NULL){
            printf("Memory allocation failed!\n");
            return 1;
        }
        MPI_Recv(localBTranspose, localOrder[1] * matrixOrder, MPI_DOUBLE, 0, 0, cartComm, MPI_STATUS_IGNORE);
    }

    // Intialize local A Flat and local B flat for rank = 0 with value from A and B^T
    if(rank == 0){
        localA = malloc(localOrder[0] * matrixOrder * sizeof(double));
        if(localA == NULL){
            printf("Memory allocation failed!\n");
            return 1;
        }
        localBTranspose = malloc(localOrder[1] * matrixOrder * sizeof(double));
        if(localBTranspose == NULL){
            printf("Memory allocation failed!\n");
            return 1;
        }
        for(int i = 0; i < localOrder[0]; i++){
            for(int j = 0; j < matrixOrder; j++){
                localA[i * matrixOrder + j] = A[i * matrixOrder + j];
            }
        }

        for(int i = 0; i < localOrder[1]; i++){
            for(int j = 0; j < matrixOrder; j++){
                localBTranspose[i * matrixOrder + j] = BTranspose[i * matrixOrder + j];
            }
        }
    }

    int step = 0;
    // Allocate the memory for the local C  matrix
    double *localC = calloc(localOrder[0] * localOrder[1], sizeof(double));
    if(localC == NULL){
        printf("Memory allocation failed!\n");
        return 1;
    }
    // Allocate the memory for M matrix used for the shift of A inside the process will be executed
    double *localMMatrix = malloc(localOrder[0] * localOrder[1] * sizeof(double));
    if(localMMatrix == NULL){
        printf("Memory allocation failed!\n");
        return 1;
    }
    // Allocate the memory for N matrix used for the shift of B inside the process will be executed
    double *localNMatrix = malloc(localOrder[0] * localOrder[1] * sizeof(double));
    if(localNMatrix == NULL){
        printf("Memory allocation failed!\n");
        return 1;
    }
    // Allocate the memory for the output P from A shift
    double *localP = malloc(localOrder[0] * sizeof(double));
    if(localP == NULL){
        printf("Memory allocation failed!\n");
        return 1;
    }
    // Allocate the memory for the output Q from B shift
    double *localQ = malloc(localOrder[1] * sizeof(double));
    if(localQ == NULL){
        printf("Memory allocation failed!\n");
        return 1;
    }
    // Allocate the memory for the input from A shift
    double *localNewMColumn = malloc(localOrder[0] * sizeof(double));
    if(localNewMColumn == NULL){
        printf("Memory allocation failed!\n");
        return 1;
    }
    // Allocate the memory for the input from B shift
    double *localNewNRow = malloc(localOrder[1] * sizeof(double));
    if(localNewNRow == NULL){
        printf("Memory allocation failed!\n");
        return 1;
    }

    // Initialize submatrices with 0 values
    for(int i = 0; i < localOrder[0]; i++){
        for(int j = 0; j < localOrder[1]; j++){
            localC[i * localOrder[1] + j] = 0;
            localMMatrix[i * localOrder[1] + j] = 0;
            localNMatrix[i * localOrder[1] + j] = 0;
        }
    }

    // The number of iterations for the systolic method are 3 * matrix order - 1
    while(step < 3 * matrixOrder - 1){
        // Update the column to receive for the processes on the first column
        if(actualprocessCoords[1] == 0){
            for(int i=0;i<localOrder[0];i++)
                localNewMColumn[i] = (i + baseMatrixCoords[0] <= step && (step - i - baseMatrixCoords[0])<matrixOrder) ? localA[i*matrixOrder + step-i- baseMatrixCoords[0]] : 0;
        }

        // Update the row to receive for the processes on the first row
        if(actualprocessCoords[0] == 0){
            for(int j=0;j<localOrder[1];j++)
                localNewNRow[j] = (j + baseMatrixCoords[1] <= step && (step - j - baseMatrixCoords[1])<matrixOrder) ? localBTranspose[j*matrixOrder + (step-j - baseMatrixCoords[1])] : 0;
        }


        // Update the column to send
        if(actualprocessCoords[1] != processOrganizationDims[1] - 1){
            for(int i = 0; i < localOrder[0]; i++){
                localP[i] = localMMatrix[i * localOrder[1] + localOrder[1] - 1];
            }
        }

        // Update the row to send
        if(actualprocessCoords[0] != processOrganizationDims[0] - 1){
            for(int j = 0; j < localOrder[1]; j++){
                localQ[j] = localNMatrix[(localOrder[0] - 1) * localOrder[1] + j];
            }
        }   

        int upRank, downRank;
        int leftRank, rightRank;

        // vertical neighbors (dim 0)
        MPI_Cart_shift(cartComm, 0, 1, &upRank, &downRank);

        // horizontal neighbors (dim 1)
        MPI_Cart_shift(cartComm, 1, 1, &leftRank, &rightRank);

        // Horizontal shift: M matrix
        if(leftRank != MPI_PROC_NULL || rightRank != MPI_PROC_NULL) {
            MPI_Sendrecv(
                    localP, localOrder[0], MPI_DOUBLE, rightRank, 0,
                    localNewMColumn, localOrder[0], MPI_DOUBLE, leftRank, 0,
                    cartComm, MPI_STATUS_IGNORE
                );
        }

        // Vertical shift: N matrix
        if(upRank != MPI_PROC_NULL || downRank != MPI_PROC_NULL) {
            MPI_Sendrecv(
                    localQ, localOrder[1], MPI_DOUBLE, downRank, 1,
                    localNewNRow, localOrder[1], MPI_DOUBLE, upRank, 1,
                    cartComm, MPI_STATUS_IGNORE
                );
        }

        // Shift M Matrix to the right (columns)
        // iterate from right to left
        for(int i = 0; i < localOrder[0]; i++){
            for(int j = localOrder[1]-1; j > 0; j--){
                localMMatrix[i * localOrder[1] + j] = localMMatrix[i * localOrder[1] + j - 1];
            }
        }

        // Shift N Matrix down (rows)
        // iterate from bottom to top
        for(int i = localOrder[0]-1; i > 0; i--){
            for(int j = 0; j < localOrder[1]; j++){
                localNMatrix[i * localOrder[1] + j] = localNMatrix[(i-1) * localOrder[1] + j];
            }
        }

        // Put the new column into the M Matrix
        for(int i = 0; i < localOrder[0]; i++){
            localMMatrix[i * localOrder[1]] = localNewMColumn[i];
        }

        // Put the new row into the N Matrix
        for(int j = 0; j < localOrder[1]; j++){
            localNMatrix[j] = localNewNRow[j];
        }

        // Update C local
        for(int i = 0; i < localOrder[0]; i++){
            for(int j = 0; j < localOrder[1]; j++){
                localC[i * localOrder[1] + j] += localMMatrix[i * localOrder[1] + j] * localNMatrix[i * localOrder[1] + j];
            }
        }

        // Update the step
        step++;

    }


    if(rank != 0){
        MPI_Send(localC, localOrder[0] * localOrder[1], MPI_DOUBLE, 0, 0, cartComm);
    }
    else{
        C = malloc(matrixOrder * matrixOrder * sizeof(double));
        if(C == NULL){
            printf("Memory allocation failed!\n");
            return 1;
        }
        
        for(int i = 0; i < localOrder[0]; i++){
            for(int j = 0; j < localOrder[1]; j++){
                C[i * matrixOrder + j] = localC[i * localOrder[1] + j];
            }
        }
        double * buffer;
        for(int p=1; p < size; p++){
            int processsCoordsRecv[2] = {0, 0};
            MPI_Cart_coords(cartComm, p, 2, processsCoordsRecv);


            int baseMatrixCoordsRecv[2] = {
                                    getBaseMatrixJCoord(matrixOrder, processOrganizationDims[0], remainder[0], processsCoordsRecv[0]), 
                                    getBaseMatrixJCoord(matrixOrder, processOrganizationDims[1], remainder[1], processsCoordsRecv[1])
                                    };

            int localOrderRecv[2] = {
                                    matrixOrder / processOrganizationDims[0] + (processsCoordsRecv[0] < (remainder[0]) ? 1 : 0),
                                    matrixOrder / processOrganizationDims[1] + (processsCoordsRecv[1] < (remainder[1]) ? 1 : 0)
                                    };

            buffer = malloc(localOrderRecv[0] * localOrderRecv[1] * sizeof(double));
            if(buffer == NULL){
                printf("Memory allocation failed!\n");
                return 1;
            }
            
            MPI_Recv(buffer, localOrderRecv[0] * localOrderRecv[1], MPI_DOUBLE, p, 0, cartComm, MPI_STATUS_IGNORE);
            for(int i = 0; i < localOrderRecv[0]; i++){
                for(int j = 0; j < localOrderRecv[1]; j++){
                    C[(baseMatrixCoordsRecv[0] + i) * matrixOrder + baseMatrixCoordsRecv[1] + j] = buffer[i * localOrderRecv[1] + j];
                }
            }  
            free(buffer);
        }
    }

    MPI_Barrier(cartComm);
    end = MPI_Wtime();


    if(rank == 0){
        printf("Order: %d, Processes: %d, Execution time: %f s\n", matrixOrder, size, end - start);
        saveMatrices(C, matrixOrder, OUTPUTFILE);
    }

    // Free root buffers
    if(rank == 0){
        free(A);
        free(BTranspose);
    }

    // Free the allocated memory
    free(localC);
    free(localMMatrix);
    free(localNMatrix);
    free(localP);
    free(localQ);
    free(localNewMColumn);
    free(localNewNRow);
    if(localA) free(localA);
    if(localBTranspose) free(localBTranspose);
    if(rank == 0 && C) free(C);

    MPI_Finalize();

    return 0;
}