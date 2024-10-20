#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define MAT_SIZE 500
#define ROOT 0

void printArray(double* mat, int n, int m){
    for (int i = 0; i<n;i++){
        for(int j=0;j<m;j++){
            printf("%f ", mat[i*m + j]);
        }
        printf("\n");
    }
}

void brute_force_matmul(double* mat1, double* mat2, double* res) {
    for (int i = 0; i < MAT_SIZE; ++i) {
        for (int j = 0; j < MAT_SIZE; ++j) {
            res[i * MAT_SIZE + j] = 0;
            for (int k = 0; k < MAT_SIZE; ++k) {
                res[i * MAT_SIZE + j] += mat1[i * MAT_SIZE + k] * mat2[k * MAT_SIZE + j];
            }
        }
    }
}

int checkRes(const double* target, const double* res) {
    for (int i = 0; i < MAT_SIZE; ++i) {
        for (int j = 0; j < MAT_SIZE; ++j) {
            if (res[i * MAT_SIZE + j] != target[i * MAT_SIZE + j]) {
                return 0;
            }
        }
    }
    return 1;
}

int main(int argc, char *argv[])
{
    int rank, mpiSize;
    double *a, *b, *c, *bfRes, *localA, *localC;
    int row_per_process;

    /* Initialize MPI */
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    /* divide how to split data */
    if (MAT_SIZE % mpiSize == 0) {
        row_per_process = MAT_SIZE / mpiSize;
    } else {
        if (rank == ROOT)
            printf("Matrix size of %d cannot be equally distributed over %d processes!\n", MAT_SIZE, mpiSize);
        MPI_Finalize();
        return 0;
    }

    /* Allocate matrices in contiguous memory */
    a = (double*) malloc(MAT_SIZE * MAT_SIZE * sizeof(double));
    b = (double*) malloc(MAT_SIZE * MAT_SIZE * sizeof(double));
    c = (double*) malloc(MAT_SIZE * MAT_SIZE * sizeof(double));
    bfRes = (double*) malloc(MAT_SIZE * MAT_SIZE * sizeof(double));
    localA = (double*) malloc(row_per_process * MAT_SIZE * sizeof(double));
    localC = (double*) malloc(row_per_process * MAT_SIZE * sizeof(double));

    double start;

    if (rank == ROOT) {
        /* Fill matrices A and B */
        for (int i = 0; i < MAT_SIZE; i++) {
            for (int j = 0; j < MAT_SIZE; j++) {
                a[i * MAT_SIZE + j] = 1.0;
                b[i * MAT_SIZE + j] = 1.0;
            }
        }

        /* Brute force multiplication for comparison */
        brute_force_matmul(a, b, bfRes);

        /* Measure start time */
        start  = MPI_Wtime();

        /* Scatter matrix A to worker tasks */
        MPI_Scatter(a, row_per_process * MAT_SIZE, MPI_DOUBLE, localA, row_per_process * MAT_SIZE, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
        MPI_Bcast(b, MAT_SIZE * MAT_SIZE, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
    } else {
        /* Worker process: Receive scattered data */
        MPI_Scatter(NULL, row_per_process * MAT_SIZE, MPI_DOUBLE, localA, row_per_process * MAT_SIZE, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
        MPI_Bcast(b, MAT_SIZE * MAT_SIZE, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
    }

    /* Perform local matrix multiplication: localA * B */
    for (int i = 0; i < row_per_process; ++i) {
        for (int j = 0; j < MAT_SIZE; ++j) {
            localC[i * MAT_SIZE + j] = 0;
            for (int k = 0; k < MAT_SIZE; ++k) {
                localC[i * MAT_SIZE + j] += localA[i * MAT_SIZE + k] * b[k * MAT_SIZE + j];
            }
        }
    }

    /* Gather results from all worker tasks */
    MPI_Gather(localC, row_per_process * MAT_SIZE, MPI_DOUBLE, c, row_per_process * MAT_SIZE, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);

    if (rank == ROOT) {
        /* Measure finish time */
        double finish = MPI_Wtime();
        printf("Done in %f seconds.\n", finish - start);

//        printArray(bfRes, MAT_SIZE, MAT_SIZE);
//        printf("\n");
//        printArray(c, MAT_SIZE, MAT_SIZE);
//        printf("\n");

        if (!checkRes(bfRes, c)) {
            printf("ERROR: Calculation differs from brute force result!\n");
        } else {
            printf("Result is correct.\n");
        }
    }

    /* Free allocated memory */
    free(a);
    free(b);
    free(c);
    free(bfRes);
    free(localA);
    free(localC);

    /* Finalize MPI */
    MPI_Finalize();
    return 0;
}
