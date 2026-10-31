#include <stdio.h>
#include <omp.h>

int main(){

omp_set_num_threads(4);

printf("Static Schedule\n");

#pragma omp parallel for schedule(static)

for(int i=0;i<16;i++)
    printf("Iteration %d Thread %d\n",i,omp_get_thread_num());

printf("\nDynamic Schedule\n");

#pragma omp parallel for schedule(dynamic)

for(int i=0;i<16;i++)
    printf("Iteration %d Thread %d\n",i,omp_get_thread_num());

printf("\nGuided Schedule\n");

#pragma omp parallel for schedule(guided)

for(int i=0;i<16;i++)
    printf("Iteration %d Thread %d\n",i,omp_get_thread_num());

return 0;

}
