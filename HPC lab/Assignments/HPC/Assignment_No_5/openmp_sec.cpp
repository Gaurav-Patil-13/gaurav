#include <iostream>
#include <vector>
#include <omp.h>
using namespace std;

int main() {
    const int N = 1000000;
    vector<int> a(N);
    for (int i = 0; i < N; i++) a[i] = (i % 1000) + 1;

    long long sum = 0;
    int mx = a[0], mn = a[0];
    double avg = 0.0;

    double start = omp_get_wtime();

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            long long s = 0;
            for (int x : a) s += x;
            sum = s;
        }

        #pragma omp section
        {
            int m = a[0];
            for (int x : a) if (x > m) m = x;
            mx = m;
        }

        #pragma omp section
        {
            int m = a[0];
            for (int x : a) if (x < m) m = x;
            mn = m;
        }

        #pragma omp section
        {
            long long s = 0;
            for (int x : a) s += x;
            avg = (double)s / N;
        }
    }

    double end = omp_get_wtime();

    cout << "Sum = " << sum << endl;
    cout << "Maximum = " << mx << endl;
    cout << "Minimum = " << mn << endl;
    cout << "Average = " << avg << endl;
    cout << "OpenMP Sections Time = " << (end-start)*1000 << " ms" << endl;
}
