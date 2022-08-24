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

/**
function to read information from .txt file
filePath: path of the txt file
keyWord: keyWord used to find the line that includes the keyword
outputMat: the matrix that contains the information read from .txt file
*/
void txtRead(const string filePath, const string keyWord, Mat& outputMat){
    double d;
    ifstream inFile;
    string path = kitti360 + filePath;
    string::size_type idx;
    string target_string;
    int strStart;
    int shape1;
    int shape2;

    //"P_rect_00" is keyword for intrinsic matrix, which is 3*4 shape
    if(keyWord=="P_rect_00"){
        strStart = keyWord.length()+2;
        shape1 = 3;
        shape2 = 4;
    }
    else{
        strStart = keyWord.length();
        shape1 = 4;
        shape2 = 4;
    }

    //open txt file
    inFile.open(path);
    if(!inFile){
        cout<<"unable to open file: "<<path <<endl;
        exit(1);
    }

    //go through the txt file line by line until the keyword is found in a line
    while(!inFile.eof()){
        string inLine;
        getline(inFile, inLine,'\n');
        idx=inLine.rfind(keyWord,0);
        if(idx!=string::npos){
            target_string=inLine.substr(strStart);
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
        txtMat(Range(0,3),Range(0,3)).copyTo(outputMat);
    }

    else{
        txtMat.copyTo(outputMat);
    }
    
    // cout << keyWord<<outputMat << endl;

    //close the file
    inFile.close();
}

//read all the ply files and save the points into a big matrix
void readPLY(Mat& pcMat, string str){
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
    // string pc_path=kitti360+"data_3d_semantics/train/2013_05_28_drive_0007_sync/"+str;
    string pc_path=kitti360+"data_3d_semantics/train/2013_05_28_drive_0010_sync/"+str;
    vector<cv::String> fn;

    //read all the file names in the folder
    glob(pc_path, fn, false);
    size_t count = fn.size();
    
    //go through the files one by one
    for (int i=0; i<count; i++){
    // for (int i=0; i<2; i++){
        if (pcl::io::loadPLYFile<pcl::PointXYZ> (fn[i], *cloud) == -1) //* load the file
        {
            PCL_ERROR ("Couldn't read file %s.ply \n",fn[i]);
            exit(-1);
        }
        //go through the points one by one
        for (size_t i = 0; i < cloud->points.size (); ++i){
        // for (size_t i = 0; i < 50000; ++i){
            Mat point_mat = (Mat_<double>(1,6) << cloud->points[i].x, cloud->points[i].y, cloud->points[i].z, 
                            cloud->points[i].x, cloud->points[i].y, cloud->points[i].z);
                            // double(cloud->points[i].b), double(cloud->points[i].g), double(cloud->points[i].r));
                            
                            
            pcMat.push_back(point_mat);
        }
    }
}


bool world2image(const Mat pcMatclone, Mat& Pimggg, const Mat intrinsicMat, const int frameId){
    bool flag;
    clock_t start, end; //timing
    double t_diff; //timing
    //read cam2world matrix
    string kw = to_string(frameId);
    start=clock();
    Mat cam2world;
    txtRead("data_poses/2013_05_28_drive_0010_sync/cam0_to_world.txt",kw, cam2world);
    Mat R;
    cam2world(Range(0,3),Range(0,3)).copyTo(R);
    // R = cam2world(Range(0,3),Range(0,3));
    Mat t;
    cam2world(Range(0,3),Range(3,4)).copyTo(t);
    // t = cam2world(Range(0,3),Range(3,4));
    // cout<<"cam2world: "<<cam2world<<endl;

    //convert from world frame to camera frame
    Mat Pworld;
    // pcMatclone.rowRange(0,3).copyTo(Pworld);
    Pworld = pcMatclone.rowRange(0,3);
    Pworld.row(0)-=t.at<double>(0,0);//p-t
    Pworld.row(1)-=t.at<double>(1,0);//p-t
    Pworld.row(2)-=t.at<double>(2,0);//p-t
    Mat point_cam = R.t() * Pworld; // R.inv() = R.t(), world frame convert to camera frame
    // Mat Pcam = Pworld;
    // Pcam = R.t() * Pworld;

    //convert from camera frame to image frame
    Mat point_proj = intrinsicMat * point_cam; //intrinsic matrix * point_cam = point in image frame
    // Mat Pproj = Pcam;
    // Pproj = intrinsicMat * Pcam;

    point_proj.copyTo(pcMatclone.rowRange(0,3));
    

    /////////////////////////////extract points with positive depth in camera frame///////////////
    
    pcMatclone.convertTo(pcMatclone, CV_64FC1);
    if (pcMatclone.type()!=6){
        cout<<"line 206, pcMatclone'type not converted to CV_64F"<<endl;
        exit(1);
    }
    // cout<<pcMatclone.row(2).colRange(0,20);
    end = clock(); //stop timing
    t_diff=(double)(end-start)/CLOCKS_PER_SEC; //calculate time difference
    // printf("time for phase 1 %f \n", t_diff);
    //I'm looking for a way to get rid of columns with d<=0, where d is the elements in the third row of a 3xN matrix.
    // for(int i=0; i<pcMatclone.cols; i++){
    //     if(pcMatclone.at<double>(2,i)>0){
    //         filter1.push_back(pcMatclone.col(i).t());
    //     }
    // }

    // filter1 = filter1.t();
    // cout<<pcMatclone.size<<endl;
    // cout<<pcMatclone.col(1)<<endl;
    Mat filter1(pcMatclone.rows, pcMatclone.cols, pcMatclone.type());
    int non_zeros=0;
    // vector<int> non_zero_idx;
    start = clock();
    for(int i=0; i<pcMatclone.cols; i++){
        if(pcMatclone.at<double>(2,i)>0){
            pcMatclone.col(i).copyTo(filter1.col(non_zeros));
            // filter1.col(non_zeros) = pcMatclone.col(i);
            non_zeros+=1;
        }
    }
    // cout<<filter1.row(0).colRange(0,20)<<endl;
    // cout<<filter1.row(3).colRange(0,20)<<endl;
    // cout<<filter1.col(1)<<endl;
    end = clock(); //stop timing
    t_diff=(double)(end-start)/CLOCKS_PER_SEC; //calculate time difference
    // printf("first big for loop %f \n", t_diff);
    Mat filter2;
    // cout<<"non_zeros"<<non_zeros<<endl;
    if(non_zeros==0){
        flag=false;
        return flag;
    }

    filter2 = filter1.colRange(0,non_zeros);
    // cout<<"points infront of camera: "<<filter1.size<<endl;
    
    //////////////////////////normalise image coordinate//////////////////////////////
    start=clock();
    // cout<<non_zeros<<endl;
    // filter1.colRange(0,non_zeros).copyTo(Pimg);
    filter2.copyTo(Pimggg);
    // cout<<"this is passed"<<endl;
    // cout<<Pimg.size<<endl;
    // cout<<Pimg.dims<<endl;
    // Pimg = filter1.colRange(0,non_zeros).clone();
    Pimggg.row(0) = filter2.row(0)/filter2.row(2);
    Pimggg.row(1) = filter2.row(1)/filter2.row(2);
    end = clock(); //stop timing
    t_diff=(double)(end-start)/CLOCKS_PER_SEC; //calculate time difference
    // printf("time for phase3 %f \n", t_diff);
    flag=true;
    return flag;



}


int main(){
    int valid_frames =0;
    int skipped_frames=0;
    int noProj_frames=0;
    int imgW=1408; //img width
    int imgH=376;   //img height
    int tempWl = 200;   //template width
    int tempWr = 1200;
    int tempHu = 80;   //template hight
    int tempHd = 300;
    int frameId; //the index of frame
    ifstream inFile; //used to read txt files
    clock_t start, end; //timing
    double t_diff; //timing
    double val_min=0, val_max=0; //max and min value within a matrix
    Mat intrinsicMat; // intrinsic matrix
    Mat pcMat; //matrix that stores pointcloud information
    Mat histMat = Mat::ones(300,300,CV_32SC1); //big histogram matrix
    int histSum=0; //sum of all the values in histMat
    vector<uchar> seg_class; //segmentation classes
    bool flag; //flag indicating if there are valid points projected to a certain frame


    //read matrices from files
    readPLY(pcMat, "static"); //read static ply files
    // readPLY(pcMat, "dynamic"); //read dynamic ply files
    pcMat = pcMat.t(); //transpose pcMat
    // cout<<"pc_Mat size "<<pcMat.size<<endl;
    txtRead("calibration/perspective.txt", "P_rect_00", intrinsicMat);//read intrinsic matrix


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


    //open cam0_to_world file and extract all the frame index
    inFile.open(kitti360+"data_poses/2013_05_28_drive_0010_sync/cam0_to_world.txt");
    if(!inFile){
        cout<<"unable to open file: cam0_to_world.txt"<<endl;
        exit(1);
    }

    vector<int> frameSequence;
    while(!inFile.eof()){
        string inLine;
        getline(inFile, inLine,'\n');
        stringstream strs(inLine);
        strs>>frameId; //the first 
        frameSequence.push_back(frameId);

    }
    inFile.close();


    //get all the picture names in segmentation folder
    string seg_path=kitti360+"data_2d_raw/2013_05_28_drive_0010_sync/image_00/segmentation";
    vector<cv::String> fn_seg;
    glob(seg_path, fn_seg, false);
    // size_t fn_count = fn.size();


    //loop through frame by frame
    for(int i=0; i<frameSequence.size()-1; i++){
    // for(int i=802; i<805; ++i){
        frameId = frameSequence[i];
        cout<<"frame: "<<frameId<<endl;

        Mat imgMat = Mat::zeros(tempHd-tempHu+1,tempWr-tempWl+1, CV_64FC4); //stores image to be displayed
        
        Mat pcMat_clone;
        pcMat.copyTo(pcMat_clone);
        
        Mat P_img;
        start = clock();
        flag = world2image(pcMat_clone, P_img, intrinsicMat,frameId);
        if(flag==false){
            cout<<"no point projected to frame "<< frameId<<endl;
            noProj_frames+=1;
            continue;
        }
        end = clock(); //stop timing
        t_diff=(double)(end-start)/CLOCKS_PER_SEC; //calculate time difference
        printf("time for world2image %f \n", t_diff);
        // cout<<P_img(Range(2,6),Range(1,2))<<endl;
        ///////////////////////extract points within image frame/////////////////////////////////
        start = clock();
        // cout<<P_img.row(2).colRange(0,20);
        //I'm looking for a more efficient way to do this
        for(int i=0; i<P_img.cols; i++){
            if(P_img.at<double>(0,i)>tempWl && P_img.at<double>(0,i)<tempWr){
                if(P_img.at<double>(1,i)>tempHu && P_img.at<double>(1,i)<tempHd){
                    // filter3.push_back(filter2.col(i).t());
                    int u = cvRound(P_img.at<double>(0,i));
                    int v = cvRound(P_img.at<double>(1,i));
                    //only extract points that project onto image first
                    if(imgMat.at<Vec4d>(v-tempHu,u-tempWl)[0]==0 || P_img.at<double>(2,i)<imgMat.at<Vec4d>(v-tempHu,u-tempWl)[0]){
                        P_img(Range(2,6),Range(i,i+1)).copyTo(imgMat.at<Vec4d>(v-tempHu,u-tempWl));  
                        // imgMat.at<Vec4d>(v-tempHu,u-tempWl) = P_img.rowRange(2,6).colRange(i,i+1);
                    }
                    
                }
            }
        }

        end = clock();
        t_diff=(double)(end-start)/CLOCKS_PER_SEC;
        // printf("time for big loop2 is %f \n", t_diff);

        //build correspondense between the template in frame1 and the template in frame2
        Mat frame_uv;
        Mat frame_xyz = Mat::zeros(5,imgMat.cols * imgMat.rows, CV_64FC1);
        Mat frame1_u;
        Mat frame1_v;
        Mat frame1_x;
        Mat frame1_y;
        Mat frame1_z;
        Mat frame2_u;
        Mat frame2_v;
        for(int v=0; v<imgMat.rows; v++){
            for(int u=0; u<imgMat.cols; u++){
                frame1_u.push_back(u+tempWl);
                frame1_v.push_back(v+tempHu);
                frame1_x.push_back(imgMat.at<Vec4d>(v,u)[1]);
                frame1_y.push_back(imgMat.at<Vec4d>(v,u)[2]);
                frame1_z.push_back(imgMat.at<Vec4d>(v,u)[3]);
                
            }
        }
        // cout<<"322 reached"<<endl;
        // cout<<imgMat.at<Vec4d>(1,1)<<endl;
        // cout<<imgMat.at<Vec4d>(1,1)[1]<<endl;
        frame1_x = frame1_x.t();
        frame1_x.copyTo(frame_xyz.row(0));
        // cout<<"340 reached"<<endl;
        frame1_y = frame1_y.t();
        frame1_y.copyTo(frame_xyz.row(1));
        frame1_z = frame1_z.t();
        // cout<<frame1_z.col(1).t()<<endl;
        frame1_z.copyTo(frame_xyz.row(2));
        frame1_u = frame1_u.t();
        frame1_u.copyTo(frame_xyz.row(3));
        frame1_v = frame1_v.t();
        frame1_v.copyTo(frame_xyz.row(4));
        // cout<<"333 reached"<<endl;
        Mat frame_xyzC;
        frame_xyz.copyTo(frame_xyzC);
        Mat frame1_uv;
        // cout<<"337 reached"<<endl;
        // cout<<frame_xyz.col(1).t()<<endl;
        flag = world2image(frame_xyz, frame1_uv, intrinsicMat, frameId);
        if(flag==false){
            cout<<"no point projected to frame "<< frameId<<endl;
            noProj_frames+=1;
            continue;
        }
        // cout<<"339 reached"<<endl;
        frame_uv.push_back(frame1_uv.row(0));
        frame_uv.push_back(frame1_uv.row(1));
        // cout<<"342 reached"<<endl;

        Mat frame2_uv;
        int frameId2 = frameSequence[i+1];
        flag = world2image(frame_xyzC, frame2_uv, intrinsicMat, frameId2);
        if(flag==false){
            cout<<"no point projected to frame "<< frameId<<endl;
            noProj_frames+=1;
            continue;
        }
        if(frame2_uv.cols != frame_uv.cols){
            cout<<"frame "<<frameId<<" is skipped"<<endl;
            skipped_frames+=1;
            continue;
        }
        // cout<<"347 reached"<<endl;
        frame2_uv.convertTo(frame2_uv,CV_64FC1);
        for(int i=0; i<frame2_uv.cols; i++){
            int u = cvRound(frame2_uv.at<double>(0,i));
            int v = cvRound(frame2_uv.at<double>(1,i));
            frame2_u.push_back(u);
            frame2_v.push_back(v);
        }
        // cout<<"frame2_u type "<<frame2_u.type()<<endl;
        frame_uv.convertTo(frame_uv, frame2_u.type());
        // cout<<"frame_uv size: "<<frame_uv.size<<endl;
        // cout<<"frame2_u size: "<<frame2_u.size<<endl;
        // cout<<"frame2_v size: "<<frame2_v.size<<endl;
        frame2_u= frame2_u.t();
        frame_uv.push_back(frame2_u);
        frame2_v = frame2_v.t();
        frame_uv.push_back(frame2_v);

        frame_uv.convertTo(frame_uv, CV_16UC1);

        // cout<<"frame_uv is ready"<<endl;

        //////////////////////////////////////////////////////
        Mat frame1 = imread(fn_seg[frameId], IMREAD_GRAYSCALE);
        Mat frame2 = imread(fn_seg[frameId2], IMREAD_GRAYSCALE);
        // minMaxLoc(frame1, &val_min, &val_max,NULL,NULL);
        // cout<<"val_min: "<<val_min<<endl;
        // cout<<"val_max: "<<val_max<<endl;
        frame1.convertTo(frame1, CV_8UC1);
        frame2.convertTo(frame2,CV_8UC1);

        
        for(int index=0; index<frame_uv.cols; index++){
            uchar p1 = frame1.at<uchar>(frame_uv.at<ushort>(1,index), frame_uv.at<ushort>(0,index));
            uchar p2 = frame2.at<uchar>(frame_uv.at<ushort>(3,index), frame_uv.at<ushort>(2,index));
            histMat.at<int>(p1,p2) += 1;
        }
        // cout<<"histMat(7,7)"<<histMat.at<int>(7,7)<<endl;

        FileStorage fs("histMat.xml", FileStorage::WRITE);
        fs<<"histMat"<<histMat;
        fs.release();   

        
        histSum = histSum+ frame_uv.cols;
        cout<<"histSum: "<<histSum<<endl;
        

        valid_frames = valid_frames+1;
        cout<<"valid frames: "<<valid_frames<<endl;
        cout<<"skipped frames: "<<skipped_frames<<endl;
        cout<<"empty frames: "<<noProj_frames<<endl;

        
        
        
        



        // // there are 4 channels in imgMat, we need to extract the last 3 channels
        // Mat chans_mat[4];
        // split(imgMat,chans_mat);

        // vector<Mat> chans_vec;
        // chans_vec.push_back(chans_mat[1]);
        // chans_vec.push_back(chans_mat[2]);
        // chans_vec.push_back(chans_mat[3]);
        // Mat merged_mat;
        // merge(chans_vec, merged_mat);
        // merged_mat.convertTo(merged_mat,CV_8UC3);
        // // cout<<"merged_mat channels: "<<merged_mat.channels()<<endl;
        // // cout<<"merged_mat size: "<<merged_mat.size<<endl;
        // // cout<<"merged_mat type"<<merged_mat.type()<<endl;

        // // minMaxLoc(merged_mat, &val_min, &val_max,NULL,NULL);
        // // cout<<"val_min: "<<val_min<<endl;
        // // cout<<"val_max: "<<val_max<<endl;

        // // display image
        // imshow("frame " + frameId, merged_mat);
        // waitKey(0);

    

        
    }

    ////////////////////////////////////////////////////////////////
    
    

    return 0;
}