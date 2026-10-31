#include <stdio.h>
#include <omp.h>

#define N 500

int main(){

static int A[N][N],B[N][N],C[N][N];

#pragma omp parallel for collapse(2)

for(int i=0;i<N;i++){

    for(int j=0;j<N;j++){

        C[i][j]=A[i][j]+B[i][j];

    }

}

printf("Optimization Completed\n");

return 0;

}
