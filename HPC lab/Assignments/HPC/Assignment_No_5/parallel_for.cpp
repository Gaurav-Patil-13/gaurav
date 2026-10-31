#include <iostream>
#include <vector>
#include <omp.h>
using namespace std;

int main() {
    const int N = 1000000;
    vector<int> a(N);
    for (int i = 0; i < N; i++) a[i] = (i % 1000) + 1;

    long long sum = 0;
    int mx = a[0], mn = a[0], even = 0;

    double start = omp_get_wtime();

    #pragma omp parallel for reduction(+:sum,even) reduction(max:mx) reduction(min:mn)
    for (int i = 0; i < N; i++) {
        sum += a[i];
        if (a[i] % 2 == 0) even++;
        if (a[i] > mx) mx = a[i];
        if (a[i] < mn) mn = a[i];
    }

    double end = omp_get_wtime();

    cout << "Sum = " << sum << endl;
    cout << "Maximum = " << mx << endl;
    cout << "Minimum = " << mn << endl;
    cout << "Even Count = " << even << endl;
    cout << "Parallel For Time = " << (end-start)*1000 << " ms" << endl;
}
