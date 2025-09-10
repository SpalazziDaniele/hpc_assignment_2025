#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>

#define earth_t 15
#define N 1024
#define threshold 0.001
#define max_it 10000
#define max_threads 11

int main() {
    float times[max_threads];
    float W_x=0.3;
    float W_y=0.2;

// We allocate memory space for the grid and a copy of it to do parallel calculations on 

    float** grid = malloc(N * sizeof(float*));
    float** new_grid = malloc(N * sizeof(float*));
    for (int i = 0; i < N; i++) {
        grid[i] = malloc(N * sizeof(float));
        new_grid[i] = malloc(N * sizeof(float));
    }
FILE* file_grid=fopen("grid_B_10000.csv", "w");

    for (int threads = 1; threads <= max_threads; threads++) {
        int count = 0;
        double start_time, end_time;

 // Initial conditions

        for (int x = 0; x < N; x++) {
            for (int y = 0; y < N; y++) {
            if (x<(0.75*N) && x>(0.25*N) && y<(0.75*N) && y>(0.25*N))
                  grid[x][y]=540;
            else
                grid[x][y]=earth_t;
            }
        }

// We save the cells initial T in the last case where we have 11 threads

	if (threads==11){
        for (int x = 0; x < N; x++) {
            for (int y = 0; y < N; y++) {
		fprintf(file_grid, "%f", grid[x][y]);
		if (y < N - 1) { fprintf(file_grid, ","); }}
	    fprintf(file_grid, "\n");}
	        fprintf(file_grid, "\n");}

        start_time = omp_get_wtime();
        omp_set_num_threads(threads);

        while (count < max_it) {
            float max_diff = 0.0;

            #pragma omp parallel for collapse(2) reduction(max:max_diff)
            for (int x = 0; x < N; x++) {
                for (int y = 0; y < N; y++) {
                    float temp = grid[x][y];

// Propagation law + boundary conditions

                    if (x == 0 && y == 0)
                        new_grid[x][y] = W_y*grid[1][0] + grid[0][1]*W_x;
                    else if (x == N - 1 && y == N - 1)
                        new_grid[x][y] = W_x*grid[N - 1][N - 2] + grid[N - 2][N - 1]*W_y;
                    else if (x == 0 && y == N - 1)
                        new_grid[x][y] = W_x*grid[0][N - 2] + grid[1][N - 1]*W_y;
                    else if (x == N - 1 && y == 0)
                        new_grid[x][y] = W_y*grid[N - 2][0] + grid[N - 1][1]*W_x;
                    else if (x == 0)
                        new_grid[x][y] = W_x*(grid[x][y - 1] + grid[x][y + 1]) +W_y*grid[x + 1][y];
                    else if (x == N - 1)
                        new_grid[x][y] = W_x*(grid[x][y - 1] + grid[x][y + 1]) + W_y*grid[x - 1][y];
                    else if (y == 0)
                        new_grid[x][y] = W_y*(grid[x - 1][y] + grid[x + 1][y]) + W_x*grid[x][y + 1];
                    else if (y == N - 1)
                        new_grid[x][y] = W_y*(grid[x - 1][y] + grid[x + 1][y]) + W_x*grid[x][y - 1];
                    else
                        new_grid[x][y] = W_y*(grid[x - 1][y] + grid[x + 1][y]) + W_x*(grid[x][y - 1] + grid[x][y + 1]);

// End of propagation law + boundary conditions

                    float diff = fabsf(temp - new_grid[x][y]);
                    if (diff > max_diff)
                        max_diff = diff;
                }
            }


            #pragma omp parallel for collapse(2)
            for (int x = 0; x < N; x++) {
                for (int y = 0; y < N; y++) {
                    grid[x][y] = new_grid[x][y];
                }
            }

// We save the cells T every N/10 iterations, where N is the max number of them

	if(threads==11 && count%1000==0){
            for (int x = 0; x < N; x++) {
                for (int y = 0; y < N; y++) {		
                    grid[x][y] = new_grid[x][y];
		    fprintf(file_grid, "%f", grid[x][y]);
		    if (y < N - 1) 
			 fprintf(file_grid, ","); }
	        fprintf(file_grid, "\n");
            }
	        fprintf(file_grid, "\n");}
            count++;
            if (max_diff < threshold){
		printf("max diff %f\n", max_diff);
		printf ("counter %d\n", count);
                break;}
        }

        end_time = omp_get_wtime();
        times[threads - 1] = end_time - start_time;
        printf("Threads used: %d, time: %.4f seconds\n", threads, times[threads - 1]);
    }

    for (int i = 0; i < N; i++) {
        free(grid[i]);
        free(new_grid[i]);
    }
    free(grid);
    free(new_grid);

    // We save execution times in a file

    FILE* file=fopen("execution_times_B_10000.csv", "w");
    for (int i = 0; i < max_threads; i++) {
        fprintf(file, "%f", times[i]);
        if (i < max_threads- 1) { fprintf(file, ","); }
        fprintf(file, "\n");
        }
        fclose(file);
fclose(file_grid);
    return 0;
}

