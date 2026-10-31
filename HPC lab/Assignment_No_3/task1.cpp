#include <omp.h>
#include <iostream>
#include <vector>
using namespace std;

long long sumSequential(const vector<int>& a) {
    long long sum = 0;
    for (int x : a) sum += x;
    return sum;
}

long long sumCritical(const vector<int>& a) {
    long long sum = 0;
    #pragma omp parallel for
    for (int i = 0; i < (int)a.size(); ++i) {
        long long value = a[i];
        #pragma omp critical
        sum += value;
    }
    return sum;
}

long long sumReduction(const vector<int>& a) {
    long long sum = 0;
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < (int)a.size(); ++i)
        sum += a[i];
    return sum;
}

int main() {
    const int N = 1000000;
    vector<int> a(N, 1);

    cout << "Sequential : " << sumSequential(a) << '\n';
    cout << "Critical   : " << sumCritical(a) << '\n';
    cout << "Reduction  : " << sumReduction(a) << '\n';
}
