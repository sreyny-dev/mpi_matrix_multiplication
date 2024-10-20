#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define MAT_SIZE 500
#define ROOT 0

void printArray(double* mat, int n, int m){
    for (int i = 0; i < n; i++){
        for (int j = 0; j < m; j++){
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
    int *sendcounts, *displs;

    /* Initialize MPI */
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    /* Determine row distribution */
    int row_per_process = MAT_SIZE / mpiSize;
    int remainder = MAT_SIZE % mpiSize;

    sendcounts = (int*) malloc(mpiSize * sizeof(int));
    displs = (int*) malloc(mpiSize * sizeof(int));

    int offset = 0;
    for (int i = 0; i < mpiSize; i++) {
        sendcounts[i] = (i < remainder) ? (row_per_process + 1) * MAT_SIZE : row_per_process * MAT_SIZE;
        displs[i] = offset;
        offset += sendcounts[i];
    }

    /* Allocate matrices */
    a = (double*) malloc(MAT_SIZE * MAT_SIZE * sizeof(double));
    b = (double*) malloc(MAT_SIZE * MAT_SIZE * sizeof(double));
    c = (double*) malloc(MAT_SIZE * MAT_SIZE * sizeof(double));
    bfRes = (double*) malloc(MAT_SIZE * MAT_SIZE * sizeof(double));
    localA = (double*) malloc(sendcounts[rank] * sizeof(double));
    localC = (double*) malloc(sendcounts[rank] * sizeof(double));

    double startTime;
    double endTime;
    double serialTime;
    double parallelTime;

    double start;

    if (rank == ROOT) {
        /* Fill matrices A and B */
        for (int i = 0; i < MAT_SIZE; i++) {
            for (int j = 0; j < MAT_SIZE; j++) {
                a[i * MAT_SIZE + j] = 1.0;
                b[i * MAT_SIZE + j] = 1.0;
            }
        }


        startTime= MPI_Wtime();
        /* Brute force multiplication for comparison */
        brute_force_matmul(a, b, bfRes);
        endTime= MPI_Wtime();
        serialTime = endTime-startTime;
        printf("Brute Force: %f seconds", serialTime);
        printf("\n");

        /* Measure start time */
        start  = MPI_Wtime();
    }

    /* Scatter matrix A to worker tasks using MPI_Scatterv */
    MPI_Scatterv(a, sendcounts, displs, MPI_DOUBLE, localA, sendcounts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* Broadcast matrix B to all processes */
    MPI_Bcast(b, MAT_SIZE * MAT_SIZE, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);

    /* Perform local matrix multiplication: localA * B */
    int local_rows = sendcounts[rank] / MAT_SIZE;
    for (int i = 0; i < local_rows; ++i) {
        for (int j = 0; j < MAT_SIZE; ++j) {
            localC[i * MAT_SIZE + j] = 0;
            for (int k = 0; k < MAT_SIZE; ++k) {
                localC[i * MAT_SIZE + j] += localA[i * MAT_SIZE + k] * b[k * MAT_SIZE + j];
            }
        }
    }

    /* Gather results from all worker tasks using MPI_Gatherv */
    MPI_Gatherv(localC, sendcounts[rank], MPI_DOUBLE, c, sendcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == ROOT) {
        /* Measure finish time */
        double finish = MPI_Wtime();
        parallelTime = finish - start;
        printf("Parallel: %f seconds.\n", parallelTime );
        double speed_up = serialTime/parallelTime;

        printf("Speed up (Parallel/Brute Force): %f", speed_up);
        printf("\n");

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
    free(sendcounts);
    free(displs);

    /* Finalize MPI */
    MPI_Finalize();
    return 0;
}
