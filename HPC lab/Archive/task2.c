#include<stdio.h>
#include<omp.h>

int main(){
    int arr[] = {1,2,3,4,5};
    int n = 5;

    #pragma omp parallel
    {
        #pragma omp for
        for(int i=0;i<n;i++){
            printf(" nums is %d with threadId %d\n", arr[i] , omp_get_thread_num());
        }

    }

}