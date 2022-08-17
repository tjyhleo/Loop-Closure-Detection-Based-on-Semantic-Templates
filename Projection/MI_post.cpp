#include <iostream>
#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/imgcodecs.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <set>
#include <ctime>

using namespace std;
using namespace cv;

//path to kitti360 dataset
string kitti360 = "/media/jialin/045E58135E57FC3C/UBUNTU/KITTI360/";


int main(){

    FileStorage fs_read;
    Mat result;
    Mat norm_result;
    double val_min, val_max;
    fs_read.open("MIMat.xml", FileStorage::READ);
    if(!fs_read.isOpened()){
        cout<<"MIMat.xml not opened"<<endl;
        exit(1);
    }
    fs_read["result"]>>result;
    fs_read.release();


    // minMaxLoc(result, &val_min, &val_max, NULL, NULL);
    // float mid_val = float((val_max - val_min)/2);
    // // float mid_val = 0.;
    // Mat mask0 = result<mid_val;
    // result.setTo(0,mask0);
    // Mat mask = result>mid_val;
    // normalize(result, norm_result, 0, 255, NORM_MINMAX, -1,mask);
    normalize(result, norm_result, 0, 255, NORM_MINMAX);
    norm_result.convertTo(norm_result, CV_8UC1);

    string imgName = "0000000571.png";
    string path = kitti360 + "data_2d_raw/2013_05_28_drive_0007_sync/image_00/data_rect/"+imgName;
    Mat raw_image = imread(path, IMREAD_COLOR);

    // rectangle(raw_image, Rect(0, 0, 50, 80), Scalar(255, 0, 0), 1, LINE_AA);
    for(int i=0; i<norm_result.rows; i++){
        for(int j=0; j<norm_result.cols; j++){
            if(norm_result.at<uchar>(i,j)>245){
                // rectangle(raw_image, Rect(j, i, 50, 80), Scalar(255, 0, 0), 1, LINE_AA);
                drawMarker(raw_image, Point(j,i), Scalar(0,0,255), MARKER_TILTED_CROSS, 5, 1,8);
            }
            // else if(norm_result.at<uchar>(i,j)>245){
            //     rectangle(raw_image, Rect(j, i, 50, 80), Scalar(0, 0, 255), 1, LINE_AA);
            // }
            // else{
            //     rectangle(raw_image, Rect(j, i, 50, 80), Scalar(255, 255, 0), 1, LINE_AA);
            // }
        }
    }

    imshow("matches", raw_image);

    vector<int> compression_params;
    compression_params.push_back(IMWRITE_PNG_COMPRESSION);
    compression_params.push_back(3);

    bool flag = false;
    flag = imwrite(kitti360 + "MI_images/rect_raw_"+imgName, raw_image, compression_params);

    waitKey(0);

    

    return 0;
}