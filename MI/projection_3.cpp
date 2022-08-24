#include <iostream>
#include <fstream>
#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/opencv.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/io/ply_io.h>
#include <ctime>

using namespace std;
using namespace cv;

//path to kitti360 dataset
string kitti360 = "/media/jialin/045E58135E57FC3C/UBUNTU/KITTI360/";




int main(){

    double val_min, val_max;

    // string seg_path=kitti360+"2013_05_28_drive_0007_sync_image_00/segmentation";
    string seg_path=kitti360+ "data_2d_semantics/train/2013_05_28_drive_0007_sync/image_00/semantic";
    vector<cv::String> fn_seg;
    glob(seg_path, fn_seg, false);

    for(int i=0; i<8; i++){
        Mat seg_img = imread(fn_seg[i], IMREAD_GRAYSCALE);
        cout<<"seg_img type: "<<seg_img.type()<<endl;
        minMaxLoc(seg_img, &val_min, &val_max, NULL, NULL);
        cout<<val_max<<", "<<val_min<<endl;


        //show the number of each semantic label in template
        vector<uchar> template_labels_vec;
        set<uchar> template_labels_set;
        for(int i=0; i<seg_img.rows; i++){
            for(int j=0; j<seg_img.cols; j++){
                template_labels_vec.push_back(seg_img.at<uchar>(i,j));
                template_labels_set.insert(seg_img.at<uchar>(i,j));
            }
        }
        for(set<uchar>::iterator it = template_labels_set.begin(); it!=template_labels_set.end(); it++){
            cout<<int(*it)<<" : "<<count(template_labels_vec.begin(), template_labels_vec.end(), *it)<<endl;
        }

        exit(0);
    }


    // readGT
    // readIM
    // buildhistgram
    // exporthistgram
    

    return 0;
}