#include <iostream>
#include <vector>
#include <omp.h>
using namespace std;

int main() {
    const int N = 500;
    vector<vector<double>> A(N, vector<double>(N, 1));
    vector<vector<double>> B(N, vector<double>(N, 2));
    vector<vector<double>> add(N, vector<double>(N));
    vector<vector<double>> sub(N, vector<double>(N));
    vector<vector<double>> TA(N, vector<double>(N));
    vector<vector<double>> TB(N, vector<double>(N));

    double start = omp_get_wtime();

    #pragma omp parallel sections
    {
        #pragma omp section
        for (int i=0;i<N;i++)
            for (int j=0;j<N;j++) add[i][j]=A[i][j]+B[i][j];

        #pragma omp section
        for (int i=0;i<N;i++)
            for (int j=0;j<N;j++) sub[i][j]=A[i][j]-B[i][j];

        #pragma omp section
        for (int i=0;i<N;i++)
            for (int j=0;j<N;j++) TA[j][i]=A[i][j];

        #pragma omp section
        for (int i=0;i<N;i++)
            for (int j=0;j<N;j++) TB[j][i]=B[i][j];
    }

    double end = omp_get_wtime();
    cout << "Matrix size = " << N << " x " << N << endl;
    cout << "Sections Time = " << (end-start)*1000 << " ms" << endl;
}
