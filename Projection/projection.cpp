#include <iostream>
#include <fstream>
#include <opencv4/opencv2/core.hpp>

using namespace std;
using namespace cv;

int main(){
    float f;
    double d;
    ifstream inFile;
    
    ////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////intrinsic matrix///////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////
    inFile.open("/media/jialin/045E58135E57FC3C/UBUNTU/KITTI360/calibration/perspective.txt");
    if(!inFile){
        cout<<"unable to open file" <<endl;
        exit(1);
    }

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

    
    vector<double> intrin_vec;
    stringstream ss(target_string);
    while(ss>>d){
        intrin_vec.push_back(d);
    }

    Mat intrinsic_Mat(intrin_vec);
    intrinsic_Mat = intrinsic_Mat.reshape(1,(3,4));
    // cout <<intrinsic_Mat.size<<endl;
    // Mat Intrinsic_Mat = intrinsic_Mat.colRange(1,3).clone();
    Mat Intrinsic_Mat = Mat::zeros(3,3,CV_64F);
    for (int i=0; i<3; i++){
        for (int j=0; j<3; j++){
            Intrinsic_Mat.at<double>(i,j) = intrinsic_Mat.at<double>(i,j);
        }
    }
    cout << Intrinsic_Mat << endl;

    inFile.close();


    ////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////camera_to_pose matrix///////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////
    inFile.open("/media/jialin/045E58135E57FC3C/UBUNTU/KITTI360/calibration/calib_cam_to_pose.txt");
    if(!inFile){
        cout<<"unable to open file" <<endl;
        exit(1);
    }

    // string::size_type idx2;
    kw="image_00:";
    // string target_string;
    while(!inFile.eof()){
        string inLine;
        getline(inFile, inLine,'\n');
        // strVec.push_back(inLine);
        idx=inLine.find(kw);
        if(idx!=string::npos){
            target_string=inLine.substr(10);
            break;
        }
    }
    target_string += "0 0 0 1"; 
    // cout<<target_string<<endl;
    vector<double> cam2pose_vec;
    // stringstream ss(target_string);
    ss.clear();
    ss.str(target_string);
    while(ss>>d){
        cam2pose_vec.push_back(d);
    }

    // cout<<cam2pose_vec[-1]<<endl;
    Mat cam2pose_Mat(cam2pose_vec);
    cam2pose_Mat = cam2pose_Mat.reshape(1,(4,4));
    cout << cam2pose_Mat << endl;

    inFile.close();


    //////////////////////////////////////////////////////////////////////
    ////////////////////////////////rectification_matrix//////////////////
    //////////////////////////////////////////////////////////////////////
    inFile.open("/media/jialin/045E58135E57FC3C/UBUNTU/KITTI360/calibration/perspective.txt");
    if(!inFile){
        cout<<"unable to open file" <<endl;
        exit(1);
    }

    // string::size_type idx2;
    kw="R_rect_00:";
    // string target_string;
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
    // cout<<target_string<<endl;
    vector<double> rect_vec;
    // stringstream ss(target_string);
    ss.clear();
    ss.str(target_string);
    while(ss>>d){
        rect_vec.push_back(d);
    }

    // cout<<cam2pose_vec.size()<<endl;
    Mat rect_Mat(rect_vec);
    rect_Mat = rect_Mat.reshape(1,(3,3));

    Mat Rect_Mat = Mat::eye(4,4,CV_64F);
    for (int i=0; i<3; i++){
        for (int j=0; j<3; j++){
            Rect_Mat.at<double>(i,j) = rect_Mat.at<double>(i,j);
        }
    }
    cout << Rect_Mat << endl;

    inFile.close();

    ///////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////pose//////////////////////////////////////
    inFile.open("/media/jialin/045E58135E57FC3C/UBUNTU/KITTI360/data_poses/2013_05_28_drive_0007_sync/poses.txt");
    if(!inFile){
        cout<<"unable to open file" <<endl;
        exit(1);
    }

    // string::size_type idx2;
    kw="1";
    // string target_string;
    while(!inFile.eof()){
        string inLine;
        getline(inFile, inLine,'\n');
        // strVec.push_back(inLine);
        idx=inLine.find(kw);
        if(idx!=string::npos){
            target_string=inLine.substr(2);
            break;
        }
    }
    // cout<<target_string<<endl;
    target_string += " 0 0 0 1";
    vector<double> pose_vec;
    // stringstream ss(target_string);
    ss.clear();
    ss.str(target_string);
    while(ss>>d){
        pose_vec.push_back(d);
    }

    // cout<<cam2pose_vec.size()<<endl;
    Mat pose_Mat(pose_vec);
    pose_Mat = pose_Mat.reshape(1,(4,4));

    
    cout << pose_Mat << endl;


    ////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////cam2world//////////////////////////////////////
    Mat rectInv = Rect_Mat.inv();
    // cout<<pose_Mat.size << endl;
    // cout<<cam2pose_Mat.channels()<<endl;
    // Mat cam2world = pose_Mat * cam2pose_Mat;
    Mat cam2world = (pose_Mat * cam2pose_Mat) * rectInv;

    cout<<cam2world<<endl;

    //////////////////////////////////////////////////////////////////////////////////
    ///////////////////////world2cam/////////////////////////////////////////////////////
    Mat cam2world_capped = cam2world.rowRange(0,3).clone();
    cout<<cam2world_capped<<endl;
    Mat R = cam2world_capped.colRange(0,3).clone();
    Mat t = cam2world_capped.col(3).clone();
    cout<<R.t()<<endl;
    cout<<t.t()<<endl;
    

    inFile.close();

    return 0;
}