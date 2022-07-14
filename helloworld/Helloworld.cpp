#include <iostream>
#include <vector>
#include <string>
#include "swap.h"

using namespace std;

// void swap(int &a, int &b)
// {
//     int temp;
//     temp = a;
//     a = b;
//     b = temp;
// }

int main(int argc, char **argv)
{

    int val1=10;
    int val2=20;

    cout << "before swap:" << endl;
    cout << val1 << endl;
    swap(val1, val2);
    cout << "after swap:" << endl;
    cout << val2 << endl;

    return 0;
}