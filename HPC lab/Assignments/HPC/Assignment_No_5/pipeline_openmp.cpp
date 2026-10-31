#include <iostream>
#include <vector>
#include <omp.h>
using namespace std;

int main() {
    const int N = 1000000;
    vector<double> data(N);

    for (int i = 0; i < N; i++) data[i] = i + 1;

    double sum = 0, mn = 0, mx = 0;

    double start = omp_get_wtime();

    #pragma omp parallel
    {
        #pragma omp single
        {
            #pragma omp task
            {
                // Stage 1: preprocessing
                for (int i = 0; i < N; i++)
                    data[i] = data[i] / N;
            }

            #pragma omp taskwait

            #pragma omp task
            {
                // Stage 2: computation
                for (int i = 0; i < N; i++)
                    data[i] = data[i] * data[i];
            }

            #pragma omp taskwait

            #pragma omp task
            {
                // Stage 3: postprocessing
                sum = data[0];
                mn = data[0];
                mx = data[0];

                for (int i = 1; i < N; i++) {
                    sum += data[i];
                    if (data[i] < mn) mn = data[i];
                    if (data[i] > mx) mx = data[i];
                }
            }
        }
    }

    double end = omp_get_wtime();

    cout << "Sum = " << sum << endl;
    cout << "Minimum = " << mn << endl;
    cout << "Maximum = " << mx << endl;
    cout << "Task Version Time = " << (end-start)*1000 << " ms" << endl;
}
