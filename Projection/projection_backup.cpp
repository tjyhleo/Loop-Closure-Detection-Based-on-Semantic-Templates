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

//function to read information from .txt file
void txtRead(const string filePath, const string keyWord, Mat& outputMat){
    double d;
    ifstream inFile;
    string path = kitti360 + filePath;
    string::size_type idx;
    string target_string;
    int strStart;
    int shape1;
    int shape2;
    if(keyWord=="P_rect_00"){
        strStart = keyWord.length()+2;
        shape1 = 3;
        shape2 = 4;
    }
    else if(keyWord=="image_00"){
        strStart = keyWord.length()+2;
        shape1 = 3;
        shape2 = 4;
    }
    else if(keyWord=="R_rect_00"){
        strStart = keyWord.length()+2;
        shape1 = 3;
        shape2 = 3;
    }
    else{
        strStart = keyWord.length();
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
        // outputMat = txtMat.colRange(0,3).rowRange(0,3);
        txtMat(Range(0,3),Range(0,3)).copyTo(outputMat);
    }
    else if(keyWord=="image_00"){
        Mat addedLine = (Mat_<double>(1,4) << 0,0,0,1);
        txtMat.push_back(addedLine);
        txtMat.copyTo(outputMat);
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
        txtMat.copyTo(outputMat);
    }
    
    // cout << keyWord<<outputMat << endl;

    //close the file
    inFile.close();
}

//read all the ply files and save the points into a big matrix
void readPLY(Mat& pcMat, string str){
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
    string pc_path=kitti360+"data_3d_semantics/train/2013_05_28_drive_0007_sync/"+str;
    vector<cv::String> fn;

    //read all the file names in the folder
    glob(pc_path, fn, false);
    size_t count = fn.size();
    
    //go through the files one by one
    // for (int i=0; i<count; i++){
    for (int i=0; i<2; i++){
        if (pcl::io::loadPLYFile<pcl::PointXYZRGB> (fn[i], *cloud) == -1) //* load the file
        {
            PCL_ERROR ("Couldn't read file %s.ply \n",fn[i]);
            exit(-1);
        }
        //go through the points one by one
        // for (size_t i = 0; i < cloud->points.size (); ++i){
        for (size_t i = 0; i < 10000; ++i){
            Mat point_mat = (Mat_<double>(1,6) << cloud->points[i].x, cloud->points[i].y, cloud->points[i].z, 
                            double(cloud->points[i].b),double(cloud->points[i].g),double(cloud->points[i].r));
            pcMat.push_back(point_mat);
        }
    }
}

int main(){
    int imgW=1408; //img width
    int imgH=376;   //img height
    int tempWl = 0;   //template width
    int tempWr = 1408;
    int tempHu = 0;   //template hight
    int tempHd = 376;
    int frameId; //the index of frame
    ifstream inFile; //used to read txt files
    clock_t start, end; //timing
    double t_diff; //timing
    double val_min=0, val_max=0; //max and min value within a matrix
    Mat intrinsicMat; // intrinsic matrix
    Mat cam2poseMat; //transformation matrix from camera frame to GPS frame
    Mat rectMat; //rectification matrix
    // Mat pcMat(38793945,3,CV_64FC1);
    Mat pcMat; //matrix that stores pointcloud information


    //read matrices from files
    readPLY(pcMat, "static"); //read static ply files
    readPLY(pcMat, "dynamic"); //read dynamic ply files
    pcMat = pcMat.t(); //transpose pcMat
    cout<<"pc_Mat size "<<pcMat.size<<endl;
    txtRead("calibration/perspective.txt", "P_rect_00", intrinsicMat);//read intrinsic matrix
    txtRead("calibration/calib_cam_to_pose.txt", "image_00",cam2poseMat);//read cam2poseMat
    txtRead("calibration/perspective.txt", "R_rect_00", rectMat);//read rectification matrix
    Mat rectInv = rectMat.inv(); //inverse of rectification matrix


    // 我尝试用xml文件读取和储存 pcMat，但是这样更慢
    // start=clock();
    // FileStorage fs;
    // fs.open("pointCloud.xml", FileStorage::READ);
    // if(!fs.isOpened()){
    //     cout<<"pointCloud.xml not opened"<<endl;
    //     exit(1);
    // }
    // fs["PLY"]>>pcMat;
    // fs.release();
    // end = clock(); //stop timing
    // t_diff=(double)(end-start)/CLOCKS_PER_SEC; //calculate time difference
    // printf("time to extract PLY files %f \n", t_diff);


    //open pose file
    inFile.open(kitti360+"data_poses/2013_05_28_drive_0007_sync/poses.txt");
    if(!inFile){
        cout<<"unable to open file: poses.txt"<<endl;
        exit(1);
    }


    //loop through frame by frame
    while(!inFile.eof()){
    // for(int index=0; index<1; index++){
        Mat imgMat = Mat::zeros(tempHd-tempHu+1,tempWr-tempWl+1, CV_64FC4); //stores image to be displayed

        //read pose frame by frame (since each frame has a corresponding pose)
        string inLine;
        getline(inFile, inLine,'\n');
        stringstream strs(inLine);
        strs>>frameId; //the first 
        cout<<"frameId: "<<frameId<<endl;
        string kw = to_string(frameId); //keyword used in txtRead to extract a specific line
        // string kw = "1";
        Mat poseMat; //poseMat is transformation matrix from GPS to world
        txtRead("data_poses/2013_05_28_drive_0007_sync/poses.txt",kw, poseMat); //read poseMat from pose.txt

        //////////////////////////////////world------> camera////////////////////////
        //其实这个cam2world可以直接读取，不用这么算,但再改就有点麻烦了
        Mat cam2world = (poseMat * cam2poseMat) * rectInv; //this is actually world to camera, but we will use it inversely
        Mat R;
        cam2world(Range(0,3),Range(0,3)).copyTo(R);
        Mat t;
        cam2world(Range(0,3),Range(3,4)).copyTo(t);
        // t.convertTo(t,CV_64FC1);
        // //make sure t is in correct data type
        // if (t.type()!=6){
        //     cout<<"line 185, t'type not converted to CV_64F"<<endl;
        //     exit(1);
        // }

        cout<<"cam2world: "<<cam2world<<endl;
        cout<<"R: "<<R<<endl;
        cout<<"t: "<<t<<endl;

        Mat pcMatclone;
        pcMat.copyTo(pcMatclone);
        Mat point_world;
        pcMat.rowRange(0,3).copyTo(point_world);

        point_world.row(0)-=t.at<double>(0,0);//p-t
        point_world.row(1)-=t.at<double>(1,0);//p-t
        point_world.row(2)-=t.at<double>(2,0);//p-t

        Mat point_cam = R.t() * point_world; // R.inv() = R.t(), world frame convert to camera frame

        ///////////////////////////////////////camera--------->image///////////////////////////
        Mat point_proj = intrinsicMat * point_cam; //intrinsic matrix * point_cam = point in image frame
        cout<<"projected point size: "<<point_proj.size<<endl;

        point_proj.copyTo(pcMatclone.rowRange(0,3));

        /////////////////////////////extract points with positive depth in camera frame///////////////
        Mat filter1;
        pcMatclone.convertTo(pcMatclone, CV_64FC1);
        if (pcMatclone.type()!=6){
            cout<<"line 206, pcMatclone'type not converted to CV_64F"<<endl;
            exit(1);
        }

        start = clock();
        //I'm looking for a way to get rid of columns with d<=0, where d is the elements in the third row of a 3xN matrix.
        for(int i=0; i<pcMatclone.cols; i++){
            if(pcMatclone.at<double>(2,i)>0){
                filter1.push_back(pcMatclone.col(i).t());
            }
        }
        end = clock();
        t_diff=(double)(end-start)/CLOCKS_PER_SEC;
        printf("big for loop run time is %f \n", t_diff);
        filter1 = filter1.t();
        cout<<"points infront of camera: "<<filter1.size<<endl;
        
        //////////////////////////normalise image coordinate//////////////////////////////
        Mat filter2;
        filter1.copyTo(filter2);
        filter2.row(0) = filter1.row(0)/filter1.row(2);
        filter2.row(1) = filter1.row(1)/filter1.row(2);
        // cout<<"filter2 is ready"<<endl;
        // cout<<"filter2.size: "<<filter2.size<<endl;
        // cout<<"filter2.type: "<<filter2.type()<<endl;

        ///////////////////////extract points within image frame/////////////////////////////////
        start = clock();
        //I'm looking for a more efficient way to do this
        for(int i=0; i<filter2.cols; i++){
            if(filter2.at<double>(0,i)>tempWl && filter2.at<double>(0,i)<tempWr){
                if(filter2.at<double>(1,i)>tempHu && filter2.at<double>(1,i)<tempHd){
                    // filter3.push_back(filter2.col(i).t());
                    int u = cvRound(filter2.at<double>(0,i));
                    int v = cvRound(filter2.at<double>(1,i));
                    //only extract points that project onto image first
                    if(imgMat.at<Vec4d>(v-tempHu,u-tempWl)[0]==0 || filter2.at<double>(2,i)<imgMat.at<Vec4d>(v-tempHu,u-tempWl)[0]){
                        filter2(Range(2,6),Range(i,i+1)).copyTo(imgMat.at<Vec4d>(v-tempHu,u-tempWl));  
                    }
                }
            }
        }

        end = clock();
        t_diff=(double)(end-start)/CLOCKS_PER_SEC;
        printf("big for loop2 run time is %f \n", t_diff);
        cout<<imgMat.at<Vec4d>(1,1)<<endl;

        // cout<<"img_mat size: "<<imgMat.size<<endl;
        // minMaxLoc(imgMat, &val_min, &val_max,NULL,NULL);
        // cout<<"img_mat val_min: "<<val_min<<endl;
        // cout<<"img_mat val_max: "<<val_max<<endl;

        //there are 4 channels in imgMat, we need to extract the last 3 channels
        Mat chans_mat[4];
        split(imgMat,chans_mat);

        minMaxLoc(chans_mat[1], &val_min, &val_max,NULL,NULL);
        // cout<<"chans_mat[1] val_min: "<<val_min<<endl;
        // cout<<"chans_mat[1] val_max: "<<val_max<<endl;

        vector<Mat> chans_vec;
        chans_vec.push_back(chans_mat[1]);
        chans_vec.push_back(chans_mat[2]);
        chans_vec.push_back(chans_mat[3]);
        Mat merged_mat;
        merge(chans_vec, merged_mat);
        merged_mat.convertTo(merged_mat,CV_8UC3);
        cout<<"merged_mat channels: "<<merged_mat.channels()<<endl;
        cout<<"merged_mat size: "<<merged_mat.size<<endl;
        cout<<"merged_mat type"<<merged_mat.type()<<endl;

        minMaxLoc(merged_mat, &val_min, &val_max,NULL,NULL);
        // cout<<"val_min: "<<val_min<<endl;
        // cout<<"val_max: "<<val_max<<endl;

        //display image
        imshow("frame " + frameId, merged_mat);
        waitKey(0);
    }

    inFile.close();

    return 0;
}