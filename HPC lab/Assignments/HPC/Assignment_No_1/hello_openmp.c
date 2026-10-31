#include<stdio.h>
#include<omp.h>

int main()
{
#pragma omp parallel
{
printf("Hello Thread %d\n",omp_get_thread_num());
}

return 0;
}
