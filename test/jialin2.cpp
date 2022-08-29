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

void trying(Mat testMatttt, Mat a){
	// testMatttt.row(1) = testMatttt.row(2);
	// a = testMatttt.row(0).clone();
	a.row(0)= testMatttt.row(0);
	// testMatttt.row(0).copyTo(a);
	// testMatttt.row(0).copyTo(a.row(0));
	// a = a-2;
	// Mat b=a;
	// b=b-2;
	cout<<"a: "<<a<<endl;
	cout<<"testMat: "<<testMatttt<<endl;

	// return a;
	// outputMat = a;
	// return outputMat;


}


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
	  {5,2,1,1}
	};
    // cout <<"array m: "<< m <<endl;
cv::Mat testMat(4,4,CV_32SC1,m);

cout << testMat << endl;

vector<vector<int>> bigVec;
for(int i=0; i<5; i++){
	for(int j=0; j<5; j++){
		vector<int> t;
		t.push_back(i);
		t.push_back(5);
		bigVec.push_back(t);
	}
}
for(int i=0; i<bigVec.size(); i++){
	cout<<bigVec[i][1]<<endl;
}


return 0;
}

