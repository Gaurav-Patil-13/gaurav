#include<stdio.h>
#include<omp.h>

int main(){

    #pragma omp parallel
    {
        printf("Hello world %d\n", omp_get_thread_num() );
    }

    return 0;
}