#include <stdio.h>
#include <omp.h>

int main()
{
    int square[1000];

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int total = omp_get_num_threads();

        #pragma omp single
        {
            printf("Total Threads = %d\n\n", total);
        }

        #pragma omp for
        for (int i = 0; i < 1000; i++)
        {
            square[i] = i * i;
            printf("Thread %d -> Square of %d = %d\n", tid, i, square[i]);
        }
    }

    return 0;
}
