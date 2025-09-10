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

//We allocate memory space for the grid and a copy of it to do parallel calculations on 

    float** grid = malloc(N * sizeof(float*));
    float** new_grid = malloc(N * sizeof(float*));
    for (int i = 0; i < N; i++) {
        grid[i] = malloc(N * sizeof(float));
        new_grid[i] = malloc(N * sizeof(float));
    }
    FILE* file_grid=fopen("grid_A_10000.csv", "w");  \\file where thw cells T will be saved

    for (int threads = 1; threads <= max_threads; threads++) {
        int count = 0;
        double start_time, end_time;

 //Initial conditions

        for (int x = 0; x < N; x++) {
            for (int y = 0; y < N; y++) {
                if (y < N / 2)
                    grid[x][y] = 250;
                else
                    grid[x][y] = earth_t;
            }
       }

//We save the cells initial T in the last case where we have 11 threads

       if (threads==11){
        for (int x = 0; x < N; x++) {
            for (int y = 0; y < N; y++) {
		fprintf(file_grid, "%f", grid[x][y]);
		if (y < N - 1)
		  fprintf(file_grid, ","); }
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

//Propagation law + boundary conditions

                    if (x == 0 && y == 0)
                        new_grid[x][y] = (grid[1][0] + grid[0][1]) / 2;
                    else if (x == N - 1 && y == N - 1)
                        new_grid[x][y] = (grid[N - 1][N - 2] + grid[N - 2][N - 1]) / 2;
                    else if (x == 0 && y == N - 1)
                        new_grid[x][y] = (grid[0][N - 2] + grid[1][N - 1]) / 2;
                    else if (x == N - 1 && y == 0)
                        new_grid[x][y] = (grid[N - 2][0] + grid[N - 1][1]) / 2;
                    else if (x == 0)
                        new_grid[x][y] = (grid[x][y - 1] + grid[x][y + 1] + grid[x + 1][y]) / 3;
                    else if (x == N - 1)
                        new_grid[x][y] = (grid[x][y - 1] + grid[x][y + 1] + grid[x - 1][y]) / 3;
                    else if (y == 0)
                        new_grid[x][y] = (grid[x - 1][y] + grid[x + 1][y] + grid[x][y + 1]) / 3;
                    else if (y == N - 1)
                        new_grid[x][y] = (grid[x - 1][y] + grid[x + 1][y] + grid[x][y - 1]) / 3;
                    else
                        new_grid[x][y] = (grid[x - 1][y] + grid[x + 1][y] + grid[x][y - 1] + grid[x][y + 1]) / 4;

//End of propagation law + boundary conditions

/*To change boundary conditions the block above can be sostituted with:

-Dirichlet B.C.:

for (int x=1; x<N-2; x++)
  for (int y=1; y<N-2; y++)
    new_grid[x][y]=0.25f*(grid[x-1][y]+
    grid[x+1][y]+grid[x][y-1]+grid[x][y+1]);

-Neumann B.C.:
int xm = (x==0)   ? 0     : x-1;
int xp = (x==N-1) ? N-1   : x+1;
int ym = (y==0)   ? 0     : y-1;
int yp = (y==N-1) ? N-1   : y+1;
new_grid[x][y] = 0.25f*(grid[xm][y]+
grid[xp][y]+grid[x][ym]+grid[x][yp]);

-Periodic B.C.:
int xm = (x+N-1)%N,;
xp=(x+1)%N; 
ym=(y+N-1)%N; 
yp=(y+1)%N;
new_grid[x][y] = 0.25f*(grid[xm][y]+
grid[xp][y]+grid[x][ym]+grid[x][yp]); */

                    float diff = fabsf(temp - new_grid[x][y]);
                    if (diff > max_diff)
                        max_diff = diff;
                }
            }


            #pragma omp parallel for collapse(2)
            for (int x = 0; x < N; x++) {
                for (int y = 0; y < N; y++) {
                    grid[x][y] = new_grid[x][y];}
                }

//We save the cells T every N/10 iterations, where N is the max number of them

	if (threads==11 && count%1000==0){
            for (int x = 0; x < N; x++) {
                for (int y = 0; y < N; y++) {
                    fprintf(file_grid, "%f", grid[x][y]);
                    if (y<N-1)
                        fprintf(file_grid, ",");}
                fprintf(file_grid, "\n");
                }
	        fprintf(file_grid, "\n");			}

            count++;
            if (max_diff < threshold){
		printf("max diff: %f\n counter %d\n" , max_diff,count);
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

//We save execution times in a file

    FILE* file=fopen("execution_times_A_10000.csv", "w");
    for (int i = 0; i < max_threads; i++) {
	fprintf(file, "%f", times[i]);
	if (i < max_threads- 1) { fprintf(file, ","); }
	fprintf(file, "\n");
	}
	fclose(file);
	fclose(file_grid);

    return 0;
}


