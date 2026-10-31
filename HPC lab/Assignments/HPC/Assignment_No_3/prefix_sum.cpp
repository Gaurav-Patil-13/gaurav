#include <omp.h>
#include <iostream>
#include <vector>
using namespace std;

void prefixSequential(const vector<long long>& a,
                      vector<long long>& p) {
    p.resize(a.size());
    if (a.empty()) return;
    p[0] = a[0];

    for (size_t i = 1; i < a.size(); ++i)
        p[i] = p[i-1] + a[i];
}

void prefixParallel(const vector<long long>& a,
                    vector<long long>& p) {
    int n = (int)a.size();
    p = a;
    if (n == 0) return;

    vector<long long> temp(n);

    for (int offset = 1; offset < n; offset *= 2) {
        #pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            temp[i] = p[i];
            if (i >= offset)
                temp[i] += p[i-offset];
        }

        p.swap(temp);
        if (offset > n/2) break;
    }
}

int main() {
    const int N = 1000000;
    vector<long long> a(N, 1), seq, par;

    prefixSequential(a, seq);
    prefixParallel(a, par);

    cout << "Sequential last = " << seq.back() << '\n';
    cout << "Parallel last   = " << par.back() << '\n';
    cout << "Correctness      = "
         << (seq == par ? "PASS" : "FAIL") << '\n';
}
