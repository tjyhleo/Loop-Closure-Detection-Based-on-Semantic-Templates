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
    //open txt file
    inFile.open(path);
    if(!inFile){
        cout<<"unable to open file: "<<path <<endl;
        exit(1);
    }
    //go through the txt file line by line until the target line is found
    while(!inFile.eof()){
        string inLine;
        getline(inFile, inLine,'\n');
        idx=inLine.rfind(keyWord,0);
        if(idx!=string::npos){
            // cout<<"idx: "<<idx<<endl;
            target_string=inLine.substr(strStart);
            // cout<<"target_string: "<<target_string<<endl;
            break;
        }
    }
    //read the target line into a vector
    vector<double> txtVec;
    stringstream ss(target_string);
    while(ss>>d){
        txtVec.push_back(d);
    }
    if(txtVec.size()==0){
        cout<<"target line not found in "<<filePath<<endl;
        exit(1);
    }
    //the txtMat is 1xN shape, convert it to ideal shape
    Mat txtMat(txtVec);
    txtMat = txtMat.reshape(1,(shape2, shape1));
    
    //further process the txtMat into a form that can be directly used
    if(keyWord=="P_rect_00"){
        outputMat = txtMat.colRange(0,3).rowRange(0,3);
    }
    else if(keyWord=="image_00"){
        Mat addedLine = (Mat_<double>(1,4) << 0,0,0,1);
        txtMat.push_back(addedLine);
        outputMat = txtMat;
    }
    else if(keyWord=="R_rect_00"){
        outputMat = Mat::eye(4,4,CV_64FC1);
        txtMat.convertTo(txtMat, CV_64FC1);
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
        Mat addedLine = (Mat_<double>(1,4) << 0,0,0,1);
        txtMat.push_back(addedLine);
        outputMat = txtMat;
    }
    
    // cout << keyWord<<outputMat << endl;

    //close the file
    inFile.close();

    return outputMat;

}

//read all the ply files and save the points into a big matrix
Mat readPLY(){
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
    string pc_path=kitti360+"data_3d_semantics/train/2013_05_28_drive_0007_sync/static";
    Mat pcMat;
    vector<cv::String> fn;
    //read all the file names in the folder
    glob(pc_path, fn, false);
    size_t count = fn.size();

    //go through the files one by one
    for (int i=0; i<count; i++){
    // for (int i=0; i<1; i++){
        if (pcl::io::loadPLYFile<pcl::PointXYZ> (fn[i], *cloud) == -1) //* load the file
        {
            PCL_ERROR ("Couldn't read file %s.ply \n",fn[i]);
            exit(-1);
        }
        //go through the points one by one
        for (size_t i = 0; i < cloud->points.size (); ++i){
        // for (size_t i = 0; i < 1; ++i){
            Mat point_mat = (Mat_<double>(1,3) << cloud->points[i].x, cloud->points[i].y, cloud->points[i].z);
            pcMat.push_back(point_mat);
        }
    }
    pcMat = pcMat.t();
    cout<<"pc_Mat size "<<pcMat.size<<endl;
    // cout<<"pc_Mat channel "<<pcMat.channels()<<endl;
    
    return pcMat;
        // end = clock();
        // t_diff=(double)(end-start)/CLOCKS_PER_SEC;
        // printf("run time is %f \n", t_diff);
}

int main(){
    // Mat intrinsicMat = txtRead("calibration/perspective.txt", "P_rect_00"); //intrinsic matrix
    double intrin[3][3]={
        {552.554261, 0, 682.049453},
        {0, 552.554261, 238.769549},
        {0,0,1}
    };
    Mat intrinsicMat(3,3,CV_64FC1, intrin);
    cout<<"intrinsicMat: "<<intrinsicMat;
    cout<<intrinsicMat.size<<endl;
    Mat cam2poseMat = txtRead("calibration/calib_cam_to_pose.txt", "image_00"); //camera frame to GPS frame
    Mat rectMat = txtRead("calibration/perspective.txt", "R_rect_00"); //rectification matrix
    Mat rectInv = rectMat.inv(); //inverse of rectification matrix
    Mat pcMat = readPLY(); //point cloud matrix
    // cout<<"pcMat"<<pcMat<<endl;

    int imgW=1408; //img width
    int imgH=376;   //img height
    int tempWl = 500;   //template width
    int tempWr = 1000;
    int tempHu = 100;   //template hight
    int tempHd = 200;
    int frameId;
    ifstream inFile;
    clock_t start, end;
    double t_diff;

    //open pose file
    inFile.open(kitti360+"data_poses/2013_05_28_drive_0007_sync/poses.txt");
    if(!inFile){
        cout<<"unable to open file: poses.txt"<<endl;
        exit(1);
    }

    //loop through frame by frame
    // while(!inFile.eof()){
    for(int index=0; index<1; index++){
        string inLine;
        getline(inFile, inLine,'\n');
        stringstream strs(inLine);
        strs>>frameId;
        cout<<"frameId: "<<frameId<<endl;
        // string kw = to_string(frameId);
        string kw = "40";
        //poseMat is transformation matrix from GPS to world
        Mat poseMat = txtRead("data_poses/2013_05_28_drive_0007_sync/poses.txt",kw);
        //////////////////////////////////world------> camera////////////////////////
        Mat cam2world = (poseMat * cam2poseMat) * rectInv; //this is actually world to camera, but we will use it inversely
        Mat R = cam2world.colRange(0,3).rowRange(0,3);
        Mat t = cam2world.col(3).rowRange(0,3);
        t.convertTo(t,CV_64FC1);
        //make sure t is in correct data type
        if (t.type()!=6){
            cout<<"line 185, t'type not converted to CV_64F"<<endl;
            exit(1);
        }
        start = clock(); //start timing

        cout<<"cam2pose: "<<cam2poseMat<<endl;
        cout<<"cam2world: "<<endl;
        cout<<cam2world<<endl;
        cout<<"R: "<<R<<endl;
        cout<<"t: "<<t<<endl;


        Mat point_world = pcMat.clone();
        // point_world - t
        point_world.row(0)-=t.at<double>(0,0);
        point_world.row(1)-=t.at<double>(1,0);
        point_world.row(2)-=t.at<double>(2,0);
        end = clock(); //stop timing
        t_diff=(double)(end-start)/CLOCKS_PER_SEC; //calculate time difference
        printf("run time is %f \n", t_diff);
        Mat point_cam = R.t() * point_world; // R.inv() = R.t(), world frame convert to camera frame
        ///////////////////////////////////////camera--------->image///////////////////////////
        Mat point_proj = intrinsicMat * point_cam; //intrinsic matrix * point_cam
        cout<<"projected point size: "<<point_proj.size<<endl;


        // cout<<"point_world -t: "<<point_world<<endl;
        cout<<"R.inv: "<<R.t()<<endl;
        // cout<<"point_cam: "<<point_cam<<endl;
        cout<<"intrinsic: "<<intrinsicMat<<endl;
        // cout<<"point_proj: "<<point_proj<<endl;





        /////////////////////////////extract points with positive depth in camera frame///////////////
        Mat filter1;
        point_proj.convertTo(point_proj, CV_64FC1);
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
        cout<<"filter2 is ready"<<endl;
        cout<<"filter2.size: "<<filter2.size<<endl;
        cout<<"filter2.type: "<<filter2.type()<<endl;
        double val_min=0, val_max=0;
        minMaxLoc(filter2, &val_min, &val_max,NULL,NULL);
        cout<<"val_min"<<val_min<<endl;
        cout<<"val_max"<<val_max<<endl;

        ///////////////////////extract points within image frame/////////////////////////////////
        Mat filter3;
        start = clock();
        ///////////////////I'm looking for a more efficient way to do this
        for(int i=0; i<filter2.cols; i++){
            if(filter2.at<double>(0,i)>tempWl && filter2.at<double>(0,i)<tempWr){
                if(filter2.at<double>(1,i)>tempHu && filter2.at<double>(1,i)<tempHd){
                    filter3.push_back(filter2.col(i).t());
                }
            }
        }
        cout<<"filter3 is ready"<<endl;
        cout<<"filter3 size: "<< filter3.size<<endl;
        ///////////////////////round up the image coordinates into integer/////////////////////
        filter3.convertTo(filter3, CV_16SC1);
        if (filter3.type()!=3){
            cout<<"line259, filter3 type not converted, filter3 type is "<<filter3.type()<<endl;
            exit(1);
        }
        
        // cout<<"filter3 after convert: "<<filter3<<endl;
        // filter3=filter3.t();
        // cout<<"filter3 shape: "<<filter3.size<<endl;
        // end = clock();
        // t_diff=(double)(end-start)/CLOCKS_PER_SEC;
        // printf("big for loop2 run time is %f \n", t_diff);
        // double val_min=0, val_max=0;
        //print out the maximum and minimum value within filter3
        minMaxLoc(filter3, &val_min, &val_max,NULL,NULL);
        cout<<"val_min"<<val_min<<endl;
        cout<<"val_max"<<val_max<<endl;


       
    }

    inFile.close();

    return 0;
}