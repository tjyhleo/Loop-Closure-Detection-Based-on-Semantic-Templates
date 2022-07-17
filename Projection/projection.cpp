#include <iostream>
#include <fstream>
#include <opencv4/opencv2/core.hpp>
// #include <opencv4/opencv2/opencv.hpp>
// #include <opencv4/opencv2/highgui/highgui.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/io/ply_io.h>

using namespace std;
using namespace cv;

Mat txtRead(const string filePath, const string keyWord){
    double d;
    Mat outputMat;
    ifstream inFile;
    string kitti360 = "/media/jialin/045E58135E57FC3C/UBUNTU/KITTI360/";
    string path = kitti360 + filePath;
    string::size_type idx;
    string target_string;
    int strStart;
    int shape1;
    int shape2;
    if(keyWord=="P_rect_00"){
        strStart = 11;
        shape1 = 3;
        shape2 = 4;
    }
    else if(keyWord=="image_00"){
        strStart = 10;
        shape1 = 3;
        shape2 = 4;
    }
    else if(keyWord=="R_rect_00"){
        strStart = 11;
        shape1 = 3;
        shape2 = 3;
    }
    else{
        strStart = 2;
        shape1 = 3;
        shape2 = 4;
    }

    inFile.open(path);
    if(!inFile){
        cout<<"unable to open file: "<<path <<endl;
        exit(1);
    }

    while(!inFile.eof()){
        string inLine;
        getline(inFile, inLine,'\n');
        idx=inLine.find(keyWord);
        if(idx!=string::npos){
            target_string=inLine.substr(strStart);
            break;
        }
    }

    vector<double> txtVec;
    stringstream ss(target_string);
    while(ss>>d){
        txtVec.push_back(d);
    }
    if(txtVec.size()==0){
        cout<<"target line not found in "<<filePath<<endl;
        exit(1);
    }

    Mat txtMat(txtVec);
    txtMat = txtMat.reshape(1,(shape2, shape1));

    if(keyWord=="P_rect_00"){
        outputMat = txtMat.colRange(0,3).rowRange(0,3);
    }
    else if(keyWord=="image_00"){
        Mat addedLine = (Mat_<double>(1,4) << 0,0,0,1);
        // Mat addedLine(1,4, CV_64F, (0,0,0,1));
        // vector<double> addedLine = {0,0,0,1};
        txtMat.push_back(addedLine);
        outputMat = txtMat;
    }
    else if(keyWord=="R_rect_00"){
        outputMat = Mat::eye(4,4,CV_64F);
        // outputMat.colRange(0,3).rowRange(0,3) = txtMat;
        for (int i=0; i<3; i++){
            for (int j=0; j<3; j++){
                outputMat.at<double>(i,j) = txtMat.at<double>(i,j);
            }
        }
    }
    else{
        // Mat addedLine = (Mat_<double>(4,1) << 0,0,0,1);
        Mat addedLine = (Mat_<double>(1,4) << 0,0,0,1);
        txtMat.push_back(addedLine);
        outputMat = txtMat;
    }
    
    cout << outputMat << endl;

    inFile.close();

    return outputMat;

}


int main(){
    Mat intrinsicMat = txtRead("calibration/perspective.txt", "P_rect_00");
    Mat cam2poseMat = txtRead("calibration/calib_cam_to_pose.txt", "image_00");
    Mat rectMat = txtRead("calibration/perspective.txt", "R_rect_00");
    Mat rectInv = rectMat.inv();
    ////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////cam2world//////////////////////////////////////
    Mat poseMat = txtRead("data_poses/2013_05_28_drive_0007_sync/poses.txt","1");
    Mat cam2world = (poseMat * cam2poseMat) * rectInv;

    cout<<cam2world<<endl;

    //////////////////////////////////////////////////////////////////////////////////
    ///////////////////////world2cam/////////////////////////////////////////////////////
    // Mat cam2world_capped = cam2world.rowRange(0,3).clone();
    // cout<<cam2world_capped<<endl;
    Mat R = cam2world.colRange(0,3).rowRange(0,3);
    Mat t = cam2world.col(3).rowRange(0,3);
    // cout<<R.t()<<endl;
    double p[3] = {3.111234,2.44444,5.22223};
    Mat point_world=Mat(3,1,CV_64F,p);
    Mat point_cam = R.t() * (point_world - t);
    Mat point_proj = intrinsicMat * point_cam;
    cout<<point_proj<<endl;

    return 0;
}