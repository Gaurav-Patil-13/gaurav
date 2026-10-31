#include <iostream>
#include <omp.h>

using namespace std;

int main()
{
    // Matrix A
    int A[3][3] = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };

    // Matrix B
    int B[3][3] = {
        {1, 0, 0},
        {0, 1, 0},
        {0, 0, 1}
    };

    // Result matrix
    int C[3][3] = {0};

    // Parallel matrix multiplication
    #pragma omp parallel for
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            C[i][j] = 0;

            for (int k = 0; k < 3; k++)
            {
                C[i][j] = C[i][j] + A[i][k] * B[k][j];
            }
        }

        cout << "Thread "
             << omp_get_thread_num()
             << " calculated row "
             << i << endl;
    }

    // Display result
    cout << "\nResult Matrix:\n";

    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            cout << C[i][j] << " ";
        }

        cout << endl;
    }

    return 0;
}
