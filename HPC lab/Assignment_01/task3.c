#include<stdio.h>
#include<omp.h>

int main(){
    int arr[] = {1,2,3,4,5};
    int n = 5;
    int sum =0;
    #pragma omp parallel
    {
        #pragma omp for reduction(+:sum)
        for(int i=0;i<n;i++){
            sum += arr[i];
           printf("Thread %d processed index %d\n", omp_get_thread_num(), i);
        }

    }
    printf("The total print %d\n", sum);

}
