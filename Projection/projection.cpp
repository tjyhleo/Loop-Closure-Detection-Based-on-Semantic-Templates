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
// /media/jialin/045E58135E57FC3C/UBUNTU/KITTI360
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
        txtMat.convertTo(txtMat, CV_64F);
        if (txtMat.type()!=6){
            cout<<"line 91, txtMat'type not converted to CV_64F"<<endl;
            exit(1);
        }
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

Mat readPLY(){
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
    string pc_path=kitti360+"data_3d_semantics/train/2013_05_28_drive_0007_sync/static";
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

    int imgW=1408;
    int imgH=376;
    int tempWl = 500;
    int tempWr = 1000;
    int tempHu = 100;
    int tempHd = 200;
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
    for(int index=0; index<1; index++){
        string inLine;
        getline(inFile, inLine,'\n');
        stringstream strs(inLine);
        strs>>frameId;
        cout<<"frameId: "<<frameId<<endl;
        string kw = inLine.substr(0,1);
        Mat poseMat = txtRead("data_poses/2013_05_28_drive_0007_sync/poses.txt",kw);
        //////////////////////////////////world------> camera////////////////////////
        Mat cam2world = (poseMat * cam2poseMat) * rectInv;
        // cout<<cam2world<<endl;
        Mat R = cam2world.colRange(0,3).rowRange(0,3);
        Mat t = cam2world.col(3).rowRange(0,3);
        t.convertTo(t,CV_64F);
        if (t.type()!=6){
            cout<<"line 185, t'type not converted to CV_64F"<<endl;
            exit(1);
        }
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
        ///////////////////////////////////////camera--------->image///////////////////////////
        Mat point_proj = intrinsicMat * point_cam;
        cout<<"projected point size: "<<point_proj.size<<endl;
        cout<<"projected point channel: "<<point_proj.channels()<<endl;
        /////////////////////////////extract points with positive depth in camera frame///////////////
        Mat filter1;
        point_proj.convertTo(point_proj, CV_64F);
        if (point_proj.type()!=6){
            cout<<"line 206, point_proj'type not converted to CV_64F"<<endl;
            exit(1);
        }
        start = clock();
        //I'm looking for a way to get rid of columns with d<=0, where d is the elements in the third row of a 3xN matrix.
        for(int i=0; i<point_proj.cols; i++){
            if(point_proj.at<double>(2,i)>0){
                filter1.push_back(point_proj.col(i).t());
            }
        }
        end = clock();
        t_diff=(double)(end-start)/CLOCKS_PER_SEC;
        printf("big for loop run time is %f \n", t_diff);
        filter1 = filter1.t();
        cout<<"points infront of camera: "<<filter1.size<<endl;
        
        //////////////////////////normalise image coordinate//////////////////////////////
        Mat filter2(2,filter1.cols, filter1.type());
        filter2.row(0) = filter1.row(0)/filter1.row(2);
        filter2.row(1) = filter1.row(1)/filter1.row(2);
        filter2.convertTo(filter2, CV_16S);
        if (filter2.type()!=3){
            cout<<"line231, filter2 type not converted"<<endl;
            exit(1);
        }
        cout<<"filter2 is ready"<<endl;

        ///////////////////////extract points within image frame/////////////////////////////////
        Mat filter3;
        start = clock();
        ///////////////////I'm looking for a more efficient way to do this
        for(int i=0; i<filter2.cols; i++){
            if(filter2.at<short>(0,i)>tempWl && filter2.at<short>(0,i)<tempWr){
                if(filter2.at<short>(1,i)>tempHu && filter2.at<short>(1,i)<tempHd){
                    filter3.push_back(filter2.col(i).t());
                }
            }
        }
        cout<<"filter3 is ready"<<endl;
        cout<<"filter3 size: "<< filter3.size<<endl;
        // filter3=filter3.t();
        // cout<<"filter3 shape: "<<filter3.size<<endl;
        // end = clock();
        // t_diff=(double)(end-start)/CLOCKS_PER_SEC;
        // printf("big for loop2 run time is %f \n", t_diff);
        double val_min, val_max;
        minMaxLoc(filter2, &val_min, &val_max,NULL,NULL);
        cout<<"val_min"<<val_min<<endl;
        cout<<"val_max"<<val_max<<endl;


       
    }

    inFile.close();

    return 0;
}