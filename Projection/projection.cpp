#include <iostream>
#include <fstream>
#include <opencv4/opencv2/core.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/io/ply_io.h>
#include <ctime>

using namespace std;
using namespace cv;

//path to kitti360 dataset
string kitti360 = "/media/jialin/045E58135E57FC3C/UBUNTU/KITTI360/";

//function to read information from .txt file
Mat txtRead(const string filePath, const string keyWord){
    double d;
    Mat outputMat; //matrix to be output
    ifstream inFile; //
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
    
    // cout << outputMat << endl;

    inFile.close();

    return outputMat;

}

Mat readPLY(){
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
    string pc_path=kitti360+"data_3d_semantics/2013_05_28_drive_0007_sync/static";
    Mat pcMat;
    vector<cv::String> fn;
    glob(pc_path, fn, false);
    size_t count = fn.size();

    for (int i=0; i<count; i++){
    // for (int i=0; i<2; i++){
        if (pcl::io::loadPLYFile<pcl::PointXYZ> (fn[i], *cloud) == -1) //* load the file
        {
            PCL_ERROR ("Couldn't read file %s.ply \n",fn[i]);
            exit(-1);
        }

        for (size_t i = 0; i < cloud->points.size (); ++i){
        // for (size_t i = 0; i < 5; ++i){
            Mat point_mat = (Mat_<double>(1,3) << cloud->points[i].x, cloud->points[i].y, cloud->points[i].z);
            pcMat.push_back(point_mat);
        }
    }
    pcMat = pcMat.t();
    cout<<"pc_Mat size "<<pcMat.size<<endl;
    cout<<"pc_Mat channel "<<pcMat.channels()<<endl;
    
    return pcMat;
        // end = clock();
        // t_diff=(double)(end-start)/CLOCKS_PER_SEC;
        // printf("run time is %f \n", t_diff);
}

int main(){
    Mat intrinsicMat = txtRead("calibration/perspective.txt", "P_rect_00");
    Mat cam2poseMat = txtRead("calibration/calib_cam_to_pose.txt", "image_00");
    Mat rectMat = txtRead("calibration/perspective.txt", "R_rect_00");
    Mat rectInv = rectMat.inv();
    Mat pcMat = readPLY();
    // Mat pcMat = (Mat_<double>(1,2,2) << 1,2,3,4);
    // cout<<pcMat<<endl;
    // cout<<pcMat.size<<endl;
    // cout<<pcMat.channels()<<endl;
    
    // Mat poseMat;
    int frameId;
    ifstream inFile;
    clock_t start, end;
    double t_diff;
    inFile.open(kitti360+"data_poses/2013_05_28_drive_0007_sync/poses.txt");
    if(!inFile){
        cout<<"unable to open file: poses.txt"<<endl;
        exit(1);
    }


    // while(!inFile.eof()){
    for(int index=0; index<5; index++){
        string inLine;
        getline(inFile, inLine,'\n');
        stringstream strs(inLine);
        strs>>frameId;
        cout<<"frameId: "<<frameId<<endl;
        string kw = inLine.substr(0,1);
        Mat poseMat = txtRead("data_poses/2013_05_28_drive_0007_sync/poses.txt",kw);
        Mat cam2world = (poseMat * cam2poseMat) * rectInv;
        // cout<<cam2world<<endl;
        Mat R = cam2world.colRange(0,3).rowRange(0,3);
        Mat t = cam2world.col(3).rowRange(0,3);
        start = clock();
        Mat point_world = pcMat.clone();
        //p-t
        point_world.row(0)-=t.at<double>(0,0);
        point_world.row(1)-=t.at<double>(1,0);
        point_world.row(2)-=t.at<double>(2,0);
        end = clock();
        t_diff=(double)(end-start)/CLOCKS_PER_SEC;
        printf("run time is %f \n", t_diff);
        Mat point_cam = R.t() * point_world;
        cout<<"point_cam size"<<point_cam.size<<endl;
        cout<<"intrinsic size"<<intrinsicMat.size<<endl;
        Mat point_proj = intrinsicMat * point_cam;
        cout<<"projected point size: "<<point_proj.size<<endl;
        cout<<"projected point channel: "<<point_proj.channels()<<endl;
        
        
       
    }

    inFile.close();

    return 0;
}