#include <iostream>
#include <fstream>
#include <opencv4/opencv2/core.hpp>

using namespace std;
using namespace cv;

int main(){
    float f;
    ifstream inFile;

    inFile.open("/media/jialin/045E58135E57FC3C/UBUNTU/KITTI360/calibration/perspective.txt");
    if(!inFile){
        cout<<"unable to open file" <<endl;
        exit(1);
    }

    // vector<string> strVec;
    string::size_type idx;
    string kw="P_rect_00:";
    string target_string;
    while(!inFile.eof()){
        string inLine;
        getline(inFile, inLine,'\n');
        // strVec.push_back(inLine);
        idx=inLine.find(kw);
        if(idx!=string::npos){
            target_string=inLine.substr(11);
            break;
        }
    }


    std::cout<<target_string<<std::endl;

    inFile.close();

    // vector<Point3f> cloudPoints;
    // cloudPoints.push_back((1196.659790039062,-1735.256347656250,143.373703002930));
    // double cloudPoints[3][3]={
    //     {1196.659790039062,-1735.256347656250,143.373703002930},
    //     {1198.843627929688,-1741.103149414062,141.547698974609},
    //     {1195.647460937500,-1749.011230468750,140.955093383789}
    // };

    // double intrinsicMatrix[3][3]={
    //     {552.554261, 0.000000, 682.049453},
    //     {0.000000, 552.554261, 238.769549},
    //     {0.000000, 0.000000, 1.000000}
    // };
    
    // double invextrinsicMatrix[4][4]={
    //     {-0.998463, 5.542730e-02, -0.000316, 1.216353e+03},
    //     {0.002711, 4.315643e-02, -0.999064, -1.648302e+03},
    //     {-0.055362, -9.975298e-01, -0.043240, 1.354970e+02},
    //     {0.000000, 0.000000, 0.000000, 1.000000 }
    // }

    // cv::Mat cloudPoints(cv::Size(5,5),CV_8UC1);


    return 0;
}