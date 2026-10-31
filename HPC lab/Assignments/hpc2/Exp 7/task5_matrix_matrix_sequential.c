#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char *argv[]) {
    int N = 500;

    if (argc > 1)
        N = atoi(argv[1]);

    double *A = malloc((size_t)N * N * sizeof(double));
    double *B = malloc((size_t)N * N * sizeof(double));
    double *C = malloc((size_t)N * N * sizeof(double));

    if (!A || !B || !C) {
        printf("Memory allocation failed.\n");
        return 1;
    }

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[(size_t)i * N + j] = (i + j) % 100 + 1;
            B[(size_t)i * N + j] = (i + j) % 100 + 1;
            C[(size_t)i * N + j] = 0.0;
        }
    }

    clock_t start = clock();

    for (int i = 0; i < N; i++) {
        for (int k = 0; k < N; k++) {
            double a = A[(size_t)i * N + k];

            for (int j = 0; j < N; j++) {
                C[(size_t)i * N + j] +=
                    a * B[(size_t)k * N + j];
            }
        }
    }

    clock_t end = clock();

    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;

    printf("Matrix Size = %d x %d\n", N, N);
    printf("Sequential Execution Time = %.6f seconds\n", elapsed);
    printf("Sample Result: C[0][0] = %.2f, C[%d][%d] = %.2f\n",
           C[0], N - 1, N - 1,
           C[(size_t)(N - 1) * N + (N - 1)]);

    free(A);
    free(B);
    free(C);

    return 0;
}
