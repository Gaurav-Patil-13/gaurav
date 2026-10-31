#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char *argv[]) {
    int N = 500;

    if (argc > 1)
        N = atoi(argv[1]);

    double *A = (double *)malloc((size_t)N * N * sizeof(double));
    double *X = (double *)malloc(N * sizeof(double));
    double *Y = (double *)malloc(N * sizeof(double));

    if (A == NULL || X == NULL || Y == NULL) {
        printf("Memory allocation failed.\n");
        return 1;
    }

    for (int i = 0; i < N; i++) {
        X[i] = 1.0;
        Y[i] = 0.0;

        for (int j = 0; j < N; j++) {
            A[(size_t)i * N + j] = (double)((i + j) % 100 + 1);
        }
    }

    clock_t start = clock();

    for (int i = 0; i < N; i++) {
        double sum = 0.0;

        for (int j = 0; j < N; j++) {
            sum += A[(size_t)i * N + j] * X[j];
        }

        Y[i] = sum;
    }

    clock_t end = clock();

    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;

    printf("Matrix Size = %d x %d\n", N, N);
    printf("Sequential Execution Time = %.6f seconds\n", elapsed);
    printf("Sample Result: Y[0] = %.2f, Y[%d] = %.2f\n",
           Y[0], N - 1, Y[N - 1]);

    free(A);
    free(X);
    free(Y);

    return 0;
}
