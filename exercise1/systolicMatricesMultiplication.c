#include "matrixutils.h"
#include <mpi.h>
#include <string.h>

// Function that allow to evaluate in the given direction the base coordinate of the matrix C stored in the local process
int getBaseMatrixJCoord(int matrixOrder, int processOrganizationDims, int remainder, int processCoords){
    int jCoord = 0;
    if(processCoords>=(remainder))
        jCoord = matrixOrder / processOrganizationDims * (processCoords-remainder)+ remainder * (matrixOrder / processOrganizationDims + 1);
    else
        jCoord = (matrixOrder / processOrganizationDims + 1) * processCoords;
    return jCoord;
}

// Function that allow to evaluate the dimension of data to send and to receive during the shift process and 
// if no data need to be sent or received the initial index will be -1 and the dimension 0
void getDimensionToSendRecv(int *initialIndex, int *dimensionData, int coordsSum, int localOrder, int step, int matrixOrder) {
    for(int i = 0; i < localOrder; i++){
                if(coordsSum+i<= step && step <= coordsSum+i+matrixOrder-1){
                    if(*initialIndex==-1)
                        *initialIndex=i; 
                    (*dimensionData)++;
                }
            }
}

int main(int argc, char** argv){
    // Check that the function receive at least 2 parameters (order of the matrices and number of repetitions)
    if(argc < 3){
        printf("No enough parameter given\n");
        return 1;
    }
    // Define the memory buffer for the output file name
    char fullOutputFileName[256];
    // Declarations of the matrices
    double *A = NULL, *BTranspose = NULL, *C = NULL;
    // Declarations of important dimensions
    int rank, size, matrixOrder = atoi(argv[1]);

    // Declarations of time dimensions for the evaluation of the execution time
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
        // Check that the repetition parameter is valid
        int rep = atoi(argv[2]);
        if(rep <= 0){
            printf("Invalid number of repetitions.\n");
            return 1;
        }

        // Compose the input and output file names
        char fullInputAFileName[256];
        snprintf(fullInputAFileName, sizeof(fullInputAFileName), "%s_%d_%d.csv", INPUTFILEA, matrixOrder, rep);

        char fullInputBFileName[256];
        snprintf(fullInputBFileName, sizeof(fullInputBFileName), "%s_%d_%d.csv", INPUTFILEB, matrixOrder, rep);

        snprintf(fullOutputFileName, sizeof(fullOutputFileName), "%s_%d_%d_%d.csv", OUTPUTFILE, matrixOrder, rep, size);

        // Check that the dimensions match
        if(matrixOrder != readMatrixFile(&A, fullInputAFileName) || matrixOrder != readTransposedMatrixFile(&BTranspose, fullInputBFileName)){
            printf("Error: Matrices dimensions do not match!\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
    }

    // Syncronize all processes before starting the systolic computation and the time evaluation
    MPI_Barrier(cartComm);
    start = MPI_Wtime();

    // Evaluate the remainder between the matrix order and the dimensions of the systolic array
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
    

    // Rank 0 process send the rows of A to the first column of PE and B columns to the first row of the PE to initialize the systolic evolution
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

    // Intialize local A and local B for rank = 0 with value from A and B^T
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

    // Initialize the step counter
    int step = 0;
    // Allocate the memory for the local C  matrix
    double *localC = calloc(localOrder[0] * localOrder[1], sizeof(double));
    if(localC == NULL){
        printf("Memory allocation failed!\n");
        return 1;
    }
    // Allocate the memory for M matrix used for the shift of A inside the process
    double *localMMatrix = calloc(localOrder[0] * localOrder[1], sizeof(double));
    if(localMMatrix == NULL){
        printf("Memory allocation failed!\n");
        return 1;
    }
    // Allocate the memory for N matrix used for the shift of B inside the process
    double *localNMatrix = calloc(localOrder[0] * localOrder[1], sizeof(double));
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

    // The number of iterations for the systolic method are 3 * matrix order - 2
    while(step < 3 * matrixOrder - 2){
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

        // Evaluate the dimensions of the column to send and to receive
        int colDimensionTosend = 0;
        int initialColIndexToSend = -1;
        getDimensionToSendRecv(&initialColIndexToSend,&colDimensionTosend,baseMatrixCoords[0]+baseMatrixCoords[1]+localOrder[1],localOrder[0],step, matrixOrder);
        int initialColIndexToReceive = -1;
        int colDimensionToReceive = 0;
        getDimensionToSendRecv(&initialColIndexToReceive, &colDimensionToReceive, baseMatrixCoords[0]+baseMatrixCoords[1], localOrder[0],step, matrixOrder);

        // Evaluate the dimensions of the row to send and to receive
        int rowDimensionTosend = 0;
        int initialRowIndexToSend = -1;
        getDimensionToSendRecv(&initialRowIndexToSend,&rowDimensionTosend,baseMatrixCoords[0]+baseMatrixCoords[1]+localOrder[0],localOrder[1],step, matrixOrder);
        int initialRowIndexToReceive = -1;
        int rowDimensionToReceive = 0;
        getDimensionToSendRecv(&initialRowIndexToReceive, &rowDimensionToReceive, baseMatrixCoords[0]+baseMatrixCoords[1], localOrder[1],step, matrixOrder);


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

        // Evaluate the ranks of the neighboring processes
        int upRank, downRank;
        int leftRank, rightRank;
        MPI_Cart_shift(cartComm, 0, 1, &upRank, &downRank);
        MPI_Cart_shift(cartComm, 1, 1, &leftRank, &rightRank);

        // Horizontal shift: M matrix across the processes
        if(leftRank != MPI_PROC_NULL || rightRank != MPI_PROC_NULL) {
                    if(initialColIndexToReceive != -1){
                        if(initialColIndexToSend != -1){
                            // Use Sendrecv to avoid deadlock when the process has to send and receive data
                            MPI_Sendrecv(
                                    &localP[initialColIndexToSend], colDimensionTosend, MPI_DOUBLE, rightRank, 0,
                                    &localNewMColumn[initialColIndexToReceive], colDimensionToReceive, MPI_DOUBLE, leftRank, 0,
                                    cartComm, MPI_STATUS_IGNORE
                                    );
                        }
                        else{
                            MPI_Recv(&localNewMColumn[initialColIndexToReceive], colDimensionToReceive, MPI_DOUBLE, leftRank, 0,
                                    cartComm, MPI_STATUS_IGNORE
                                    );
                        }
                    }
                    else{
                        if(initialColIndexToSend != -1){
                            MPI_Send(&localP[initialColIndexToSend], colDimensionTosend, MPI_DOUBLE, rightRank, 0,cartComm);
                        }
                    }
                            
        }

        // Vertical shift: N matrix across the processes
        if(upRank != MPI_PROC_NULL || downRank != MPI_PROC_NULL) {
            if(initialRowIndexToReceive != -1){
                        if(initialRowIndexToSend != -1){
                            // Use Sendrecv to avoid deadlock when the process has to send and receive data
                            MPI_Sendrecv(
                                    &localQ[initialRowIndexToSend], rowDimensionTosend, MPI_DOUBLE, downRank, 0,
                                    &localNewNRow[initialRowIndexToReceive], rowDimensionToReceive, MPI_DOUBLE, upRank, 0,
                                    cartComm, MPI_STATUS_IGNORE
                                    );
                        }
                        else{
                            MPI_Recv(&localNewNRow[initialRowIndexToReceive], rowDimensionToReceive, MPI_DOUBLE, upRank, 0,
                                    cartComm, MPI_STATUS_IGNORE
                                    );
                        }
                    }
                    else{
                        if(initialRowIndexToSend != -1){
                            MPI_Send(&localQ[initialRowIndexToSend], rowDimensionTosend, MPI_DOUBLE, downRank, 0,cartComm);
                        }
                    }
        }

        // Shift M Matrix to the right (columns)
        // iterate from right to left inside the process  
        for(int i = 0; i < localOrder[0]; i++){
            for(int j = localOrder[1]-1; j > 0; j--){
                localMMatrix[i * localOrder[1] + j] = localMMatrix[i * localOrder[1] + j - 1];
            }
        }

        // Put the new column into the M Matrix or set to 0 if no column is received
        if(initialColIndexToReceive != -1){     
            // Put the new column into the M Matrix
            for(int i = 0; i < localOrder[0]; i++){
                if(i>=initialColIndexToReceive && i<initialColIndexToReceive+colDimensionToReceive)
                    localMMatrix[i * localOrder[1]] = localNewMColumn[i];
                else
                    localMMatrix[i * localOrder[1]] = 0;
            }
        }
        else{
            for(int i = 0; i < localOrder[0]; i++){
                localMMatrix[i * localOrder[1]] = 0;
            }
        }

        // Shift N Matrix down (rows)
        // iterate from bottom to top inside the process
        for(int i = localOrder[0]-1; i > 0; i--){
            for(int j = 0; j < localOrder[1]; j++){
                localNMatrix[i * localOrder[1] + j] = localNMatrix[(i-1) * localOrder[1] + j];
            }
        }

        
        // Put the new row into the N Matrix or set to 0 if no row is received
        if(initialRowIndexToReceive != -1){     
            // Put the new row into the N Matrix
            for(int j = 0; j < localOrder[1]; j++){
                if(j>=initialRowIndexToReceive && j<initialRowIndexToReceive+rowDimensionToReceive)
                    localNMatrix[j] = localNewNRow[j];
                else
                    localNMatrix[j] = 0;
            }
        }
        else{
            for(int j = 0; j < localOrder[1]; j++){
                localNMatrix[j] = 0;
            }
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


    // Gather the local C matrices into the root process
    // Not rank 0 processes send their local C to rank 0
    if(rank != 0){
        MPI_Send(localC, localOrder[0] * localOrder[1], MPI_DOUBLE, 0, 0, cartComm);
    }
    else{
        // Rank 0 process receive the local C matrices from all the other processes and compose the final C matrix
        C = malloc(matrixOrder * matrixOrder * sizeof(double));
        if(C == NULL){
            printf("Memory allocation failed!\n");
            return 1;
        }
        
        // Copy the local C of rank 0 into the final C matrix
        for(int i = 0; i < localOrder[0]; i++){
            for(int j = 0; j < localOrder[1]; j++){
                C[i * matrixOrder + j] = localC[i * localOrder[1] + j];
            }
        }

        // Use a buffer to receive the local C matrices from other processes and copy them into the final C matrix
        double * buffer;
        for(int p=1; p < size; p++){
            int processsCoordsRecv[2] = {0, 0};
            MPI_Cart_coords(cartComm, p, 2, processsCoordsRecv);

            // Calculate for each process the base coords and the local order to understand where to put the data in the final C matrix
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

    // Synchronize all processes before ending the time evaluation
    MPI_Barrier(cartComm);
    end = MPI_Wtime();

    // Rank 0 process print the execution time and save the final C matrix
    if(rank == 0){
        printf("ExecTime: %f\n", end - start);
        saveMatrices(C, matrixOrder, fullOutputFileName);
    }

    // Rank 0 process free A, BTranspose
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