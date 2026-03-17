#include <iostream>
#include <cstdlib>
#include <chrono>
using namespace std;
using namespace chrono;

int findSum(int arr[], int n) {
    int sum = 0;
    for(int i = 0; i < n; i++)
        sum += arr[i];
    return sum;
}

int searchElement(int arr[], int n, int key) {
    for(int i = 0; i < n; i++) {
        if(arr[i] == key)
            return i;
    }
    return -1;
}

int main() {
    int n = 100000;
    int key = 50;

    int *arr = new int[n];

    for(int i = 0; i < n; i++)
        arr[i] = rand() % 100;

    auto start = high_resolution_clock::now();

    int sum = findSum(arr, n);
    int pos = searchElement(arr, n, key);

    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start);

    cout << "Sum: " << sum << endl;
    cout << "Key Position: " << pos << endl;
    cout << "Execution Time (Sequential): " << duration.count() << " ms" << endl;

    delete[] arr;
    return 0;
}

//o/p:
//Sum: 4952310
//Key Position: 2387
//Execution Time (Sequential): 12 ms
