#include <iostream>
#include <fstream>
#include <opencv4/opencv2/core.hpp>

using namespace std;
using namespace cv;

// Mat mat = Mat::eye(Size(12,12), CV_8UC1);
// FileStorage fs("vocabulary.xml", FileStorage::READ);
// Mat mat2;
// fs["vocabulary"] >> mat2;
// // fs<<"vocabulary"<<mat;
// cout<<mat2<<enl;
// fs.release();


int main(){
// Mat mat2;
// FileStorage fs2;
// fs2.open("vocabulary.xml", FileStorage::READ);
// if(!fs2.isOpened()){
//     cout<<"file not opened"<<endl;
//     exit(1);
// }
// fs2["vocabulary"]>>mat2;
// // mat2 = fs2["vocabulary"];
// cout<<mat2<<endl;

// fs2.release();

int m[4][4] = 
	{ {1, 2, 3,3},
	  {2,1,3,3},
	  {3,2,1,1},
	  {3,2,1,1}
	};
    // cout <<"array m: "<< m <<endl;
cv::Mat testMat(4,4,CV_32SC1,m);
cout << testMat << endl;

Mat histMat = Mat::ones(1,4,CV_32FC1);


// histMat += testMat.col(1).t();

// cout<<histMat<<endl;
testMat.convertTo(testMat, CV_32FC1);
cout<<testMat.col(0).t() /2.0<<endl;

Mat P1Mat = testMat.col(0).t() /2.0;
histMat +=testMat.col(0).t() /2.0;
cout<<histMat<<endl;


// Mat a(3,4,CV_64FC1);

// int A = 1000;
// for()

// Mat a = testMat.rowRange(0,2).clone();
// // a.row(0) = testMat.row(1) - testMat.row(2);
// Scalar ss = sum(testMat.col(0));
// float s = ss[0];
// cout<<s<<endl;
// Mat b;
// a.copyTo(b);
// b = a.clone();
// b.row(0)-=2;
// a.rowRange(0,2) = testMat.rowRange(0,2).clone();
// testMat.rowRange(0,2).copyTo(a.rowRange(0,2));
// testMat.row(1).copyTo(a.row(1));
// Mat a = testMat.rowRange(0,2);
// Mat b(4,4,a.type());
// // Mat c(2,4,a.type());
// Mat c = a;
// c = a*b;
// a=a*c;
// testMat.copyTo(a.rowRange(0,2));

// cout<<a<<endl;
// cout<<testMat<<endl;
// cout<<b<<endl;


return 0;
}

