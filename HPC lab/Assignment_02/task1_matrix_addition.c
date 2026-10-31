#include <stdio.h>
#include <omp.h>

#define SIZE 3

int main() {

    int A[SIZE][SIZE]={{1,2,3},{4,5,6},{7,8,9}};
    int B[SIZE][SIZE]={{9,8,7},{6,5,4},{3,2,1}};
    int C[SIZE][SIZE];

    omp_set_num_threads(4);

    #pragma omp parallel
    {
        int id=omp_get_thread_num();
        int total=omp_get_num_threads();

        printf("Thread %d of %d started\n",id,total);

        #pragma omp for
        for(int i=0;i<SIZE;i++)
            for(int j=0;j<SIZE;j++)
                C[i][j]=A[i][j]+B[i][j];
    }

    printf("\nResult Matrix\n");

    for(int i=0;i<SIZE;i++){
        for(int j=0;j<SIZE;j++)
            printf("%d ",C[i][j]);
        printf("\n");
    }

    return 0;
}
