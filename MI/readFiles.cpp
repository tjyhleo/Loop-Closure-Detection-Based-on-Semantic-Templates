#include <iostream>
#include <fstream>
#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/opencv.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/io/ply_io.h>

using namespace std;
using namespace cv;

string kitti360 = "/media/jialin/045E58135E57FC3C/UBUNTU/KITTI360/";

//read all the ply files and save the points into a big matrix
void readPLY(Mat& pcMat){
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
    string pc_path=kitti360+"data_3d_semantics/train/2013_05_28_drive_0007_sync/static";
    // Mat pcMat;
    vector<cv::String> fn;
    //read all the file names in the folder
    glob(pc_path, fn, false);
    size_t count = fn.size();
    
    //go through the files one by one
    for (int i=0; i<count; i++){
    // for (int i=0; i<2; i++){
        if (pcl::io::loadPLYFile<pcl::PointXYZRGB> (fn[i], *cloud) == -1) //* load the file
        {
            PCL_ERROR ("Couldn't read file %s.ply \n",fn[i]);
            exit(-1);
        }
        //go through the points one by one
        for (size_t i = 0; i < cloud->points.size (); ++i){
        // for (size_t i = 0; i < 1000; ++i){
            Mat point_mat = (Mat_<double>(1,6) << cloud->points[i].x, cloud->points[i].y, cloud->points[i].z, 
                            double(cloud->points[i].b),double(cloud->points[i].g),double(cloud->points[i].r));
            pcMat.push_back(point_mat);
            // cout<<"point: "<<cloud->points[i]<<endl;
            // if(uint8_t(cloud->points[i].g)!=128 || uint8_t(cloud->points[i].b)!=128 || uint8_t(cloud->points[i].r)!=128){
            //     cout<<"point: "<<cloud->points[i]<<endl;
            //     cout<<cloud->points[i].g<<endl;
            // }
        }
    }
    pcMat = pcMat.t();
    cout<<"pc_Mat size "<<pcMat.size<<endl;
}

int main(){
    Mat pcMat;
    readPLY(pcMat);

    
    // FileStorage fs("pointCloud.xml", FileStorage::WRITE);
    // fs<<"PLY"<<pcMat;
    // fs.release();

    return 0;
}