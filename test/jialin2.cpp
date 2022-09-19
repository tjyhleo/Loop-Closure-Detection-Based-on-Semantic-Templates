#include <iostream>
#include <fstream>
#include <opencv4/opencv2/core.hpp>
#include <set>
#include <numeric>

using namespace std;
using namespace cv;

// Mat mat = Mat::eye(Size(12,12), CV_8UC1);
// FileStorage fs("vocabulary.xml", FileStorage::READ);
// Mat mat2;
// fs["vocabulary"] >> mat2;
// // fs<<"vocabulary"<<mat;
// cout<<mat2<<enl;
// fs.release();

int MaxComb(Mat inMat){
    assert(inMat.type()==4 && "input mat type must be int");
    int score=0;
    if(inMat.rows==1 || inMat.cols==1){
        double val_max, val_min;
        minMaxLoc(inMat, &val_min, &val_max, NULL, NULL);
        score=int(val_max);
        return score;
    }

    int RUCorner = inMat.at<int>(0,inMat.cols-1);
    int LDCorner = inMat.at<int>(inMat.rows-1, 0);
    if(RUCorner>score){
        score=RUCorner;
    } 
    if(LDCorner>score){
        score=LDCorner;
    }
    for(int i=0; i<inMat.cols-1; i++){
        Mat innerMat = inMat.rowRange(1,inMat.rows).colRange(i+1,inMat.cols);
        int maxFromMat = MaxComb(innerMat);
        int col_score = inMat.at<int>(0,i) + maxFromMat;
        if(col_score>score){
            score=col_score;
        }
    }
    for(int i=0; i<inMat.rows-1; i++){
        Mat innerMat = inMat.rowRange(i+1,inMat.rows).colRange(1,inMat.cols);
        int maxFromMat = MaxComb(innerMat);
        int row_score = inMat.at<int>(i,0) + maxFromMat;
        if(row_score>score){
            score=row_score;
        }
    }
    return score;
}

template <typename T>
vector<size_t> sort_indexes_e(vector<T> &v)
{
    vector<size_t> idx(v.size());
    iota(idx.begin(), idx.end(), 0);
    sort(idx.begin(), idx.end(),
        [&v](size_t i1, size_t i2) {return v[i1] < v[i2]; });
    return idx;
}

void sortVec(vector<int>& inVec){
    sort(inVec.begin(), inVec.end());
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
	{ {1,2,3,3},
	  {2,1,3,3},
	  {3,3,1,1},
	  {5,2,1,2}
	};
    // cout <<"array m: "<< m <<endl;
cv::Mat testMat(4,4,CV_32SC1,m);
// cout<<testMat.type()<<endl;
assert(testMat.type()==4);

int score = MaxComb(testMat);
// cout<<score<<endl;

testMat.convertTo(testMat, CV_32FC1);
cv::Mat out;
log(testMat, out);
cout<<out<<endl;


return 0;
}

