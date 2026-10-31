#include <iostream>
#include <vector>
#include <algorithm>
#include <omp.h>
using namespace std;

void mergeParts(vector<int>& a, int l, int m, int r) {
    vector<int> temp;
    int i=l, j=m+1;
    while(i<=m && j<=r) {
        if(a[i] <= a[j]) temp.push_back(a[i++]);
        else temp.push_back(a[j++]);
    }
    while(i<=m) temp.push_back(a[i++]);
    while(j<=r) temp.push_back(a[j++]);
    for(int k=0;k<(int)temp.size();k++) a[l+k]=temp[k];
}

void mergeSortTask(vector<int>& a, int l, int r) {
    if(l >= r) return;

    int m = (l+r)/2;

    #pragma omp task shared(a) if(r-l > 1000)
    mergeSortTask(a, l, m);

    #pragma omp task shared(a) if(r-l > 1000)
    mergeSortTask(a, m+1, r);

    #pragma omp taskwait
    mergeParts(a, l, m, r);
}

int main() {
    const int N = 1000000;
    vector<int> a(N);

    for(int i=0;i<N;i++) a[i] = N-i;

    double start = omp_get_wtime();

    #pragma omp parallel
    {
        #pragma omp single
        mergeSortTask(a, 0, N-1);
    }

    double end = omp_get_wtime();

    cout << "Array sorted successfully." << endl;
    cout << "Threads = " << omp_get_max_threads() << endl;
    cout << "Execution Time = " << (end-start)*1000 << " ms" << endl;
}
