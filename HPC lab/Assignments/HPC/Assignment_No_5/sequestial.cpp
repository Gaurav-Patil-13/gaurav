#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <numeric>
using namespace std;

int main() {
    const int N = 1000000;
    vector<int> a(N);
    for (int i = 0; i < N; i++) a[i] = (i % 1000) + 1;

    auto start = chrono::high_resolution_clock::now();

    long long sum = 0;
    int mx = a[0], mn = a[0];
    for (int x : a) {
        sum += x;
        mx = max(mx, x);
        mn = min(mn, x);
    }
    double avg = (double)sum / N;

    auto end = chrono::high_resolution_clock::now();
    double time_ms = chrono::duration<double, milli>(end-start).count();

    cout << "Sum = " << sum << endl;
    cout << "Maximum = " << mx << endl;
    cout << "Minimum = " << mn << endl;
    cout << "Average = " << avg << endl;
    cout << "Sequential Time = " << time_ms << " ms" << endl;
}
