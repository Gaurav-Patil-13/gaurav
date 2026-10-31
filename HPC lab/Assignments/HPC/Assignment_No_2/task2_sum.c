#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 1000000

int main(){

    long long sum=0;

    int *A=(int*)malloc(sizeof(int)*N);

    for(int i=0;i<N;i++)
        A[i]=1;

    double start=omp_get_wtime();

    #pragma omp parallel for reduction(+:sum)
    for(int i=0;i<N;i++)
        sum+=A[i];

    double end=omp_get_wtime();

    printf("Sum = %lld\n",sum);
    printf("Time = %f seconds\n",end-start);

    free(A);

    return 0;
}
