#include <iostream>
using namespace std;

void asdf(int &a){
    a +=2;
}

int main(){
    int a=2;
    asdf(a);
    cout<<a<<endl;
    return 0;
}