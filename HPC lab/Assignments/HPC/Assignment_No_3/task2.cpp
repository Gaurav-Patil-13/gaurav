#include <omp.h>
#include <iostream>
#include <vector>
#include <cmath>
using namespace std;

double dotSequential(const vector<double>& a,
                     const vector<double>& b) {
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
        sum += a[i] * b[i];
    return sum;
}

double dotParallel(const vector<double>& a,
                   const vector<double>& b) {
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < (int)a.size(); ++i)
        sum += a[i] * b[i];
    return sum;
}

int main() {
    const int N = 1000000;
    vector<double> a(N, 1.5), b(N, 2.0);

    double s = dotSequential(a, b);
    double p = dotParallel(a, b);

    cout << "Sequential = " << s << '\n';
    cout << "Parallel   = " << p << '\n';
    cout << "Correctness = "
         << (fabs(s-p) < 1e-6 ? "PASS" : "FAIL") << '\n';
}
