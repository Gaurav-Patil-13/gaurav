#include <iostream>
#include <vector>
#include <omp.h>

using namespace std;


// --------------------------------------------------
// Sequential Dot Product
// --------------------------------------------------
double dotSequential(const vector<double>& a,
                     const vector<double>& b) {

    double sum = 0.0;

    for (int i = 0; i < (int)a.size(); ++i) {
        sum += a[i] * b[i];
    }

    return sum;
}


// --------------------------------------------------
// Dot Product using Critical
// --------------------------------------------------
double dotCritical(const vector<double>& a,
                   const vector<double>& b) {

    double sum = 0.0;

    #pragma omp parallel for
    for (int i = 0; i < (int)a.size(); ++i) {

        double value = a[i] * b[i];

        #pragma omp critical
        sum += value;
    }

    return sum;
}


// --------------------------------------------------
// Dot Product using Atomic
// --------------------------------------------------
double dotAtomic(const vector<double>& a,
                 const vector<double>& b) {

    double sum = 0.0;

    #pragma omp parallel for
    for (int i = 0; i < (int)a.size(); ++i) {

        double value = a[i] * b[i];

        #pragma omp atomic
        sum += value;
    }

    return sum;
}


// --------------------------------------------------
// Dot Product using Reduction
// --------------------------------------------------
double dotReduction(const vector<double>& a,
                    const vector<double>& b) {

    double sum = 0.0;

    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < (int)a.size(); ++i) {

        sum += a[i] * b[i];
    }

    return sum;
}


// --------------------------------------------------
// Main Function
// --------------------------------------------------
int main() {

    // Number of elements
    const int N = 1000000;

    // Create two vectors
    vector<double> a(N, 1.0);
    vector<double> b(N, 2.0);


    // --------------------------------------------------
    // Sequential
    // --------------------------------------------------

    double start = omp_get_wtime();

    double resultSequential = dotSequential(a, b);

    double end = omp_get_wtime();

    double timeSequential = end - start;


    // --------------------------------------------------
    // Critical
    // --------------------------------------------------

    start = omp_get_wtime();

    double resultCritical = dotCritical(a, b);

    end = omp_get_wtime();

    double timeCritical = end - start;


    // --------------------------------------------------
    // Atomic
    // --------------------------------------------------

    start = omp_get_wtime();

    double resultAtomic = dotAtomic(a, b);

    end = omp_get_wtime();

    double timeAtomic = end - start;


    // --------------------------------------------------
    // Reduction
    // --------------------------------------------------

    start = omp_get_wtime();

    double resultReduction = dotReduction(a, b);

    end = omp_get_wtime();

    double timeReduction = end - start;


    // --------------------------------------------------
    // Display Results
    // --------------------------------------------------

    cout << "========================================" << endl;
    cout << "       DOT PRODUCT RESULTS" << endl;
    cout << "========================================" << endl;

    cout << "\nSequential:" << endl;
    cout << "Result = " << resultSequential << endl;
    cout << "Time   = " << timeSequential << " seconds" << endl;


    cout << "\nCritical:" << endl;
    cout << "Result = " << resultCritical << endl;
    cout << "Time   = " << timeCritical << " seconds" << endl;


    cout << "\nAtomic:" << endl;
    cout << "Result = " << resultAtomic << endl;
    cout << "Time   = " << timeAtomic << " seconds" << endl;


    cout << "\nReduction:" << endl;
    cout << "Result = " << resultReduction << endl;
    cout << "Time   = " << timeReduction << " seconds" << endl;


    cout << "\n========================================" << endl;

    return 0;
}
