#include <iostream>
#include <thread>
using namespace std;

void printNumbers(int n) {
    for(int i = 1; i <= n; i++) {
        cout << i << " ";
    }
}

int main() {
    int n;
    cout << "Enter n: ";
    cin >> n;

    thread t(printNumbers, n);

    t.join();

    return 0;
}
