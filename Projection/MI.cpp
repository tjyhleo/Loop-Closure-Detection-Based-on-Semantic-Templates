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

/**
 * @brief calculate mutual information
 * 
 * @param srccc part of source image that overlaps with template image sliding window
 * @param temppp template image
 * @param histtt histogram matrix
 * @param row_sum sum of each row of histogram matrix
 * @param col_sum sum of each column of histogram matrix
 * @param total sum of histogram matrix
 * @param Pst_list stores semantic labels, corresponding Pst value and attribution to MI
 * @return float MI score
 */
float MI_calculator(const Mat temppp, const Mat srccc, const Mat histtt, const int N)
    {
        clock_t start, end; //timing
        double t_diff; //timing
        float I=0.;
        float Pst, Pt, Ps;

        for(int i=0; i<histtt.rows; ++i){
            for(int j=0; j<histtt.cols; ++j){
                Pst = float(histtt.at<float>(i,j))/N;
                Pt = float(temppp.at<float>(0,i))/N;
                Ps = float(srccc.at<float>(0,j))/N;
                
                I += Pst * log(Pst/(Ps*Pt));
            }
        }
        
        return I;

    }


void hist_builder(const Mat input_mat, const Mat input_mat2, Mat& output_mat, 
                 const vector<Mat>row_mat_vec, const Mat row_mat[20][20]){
    // assert(input_mat.size==input_mat2.size);
    uchar P1;
    uchar P2;
    if(input_mat2.empty()){
        for(int i=0; i<input_mat.rows; ++i){
            for(int j=0; j<input_mat.cols; ++j){
                P1 = input_mat.at<uchar>(i,j);
                output_mat.at<float>(0,P1) += 1;
                // output_mat += row_mat_vec[P1];
            }
        }
    }
    else{
        for(int i=0; i<input_mat.rows; ++i){
            for(int j=0; j<input_mat.cols; ++j){
                P1 = input_mat.at<uchar>(i,j);
                P2 = input_mat2.at<uchar>(i,j);
                output_mat.at<float>(P1,P2) += 1;
                // output_mat += row_mat[P1][P2];

            }
        }
    }

}

bool templateFinder(Mat img, Mat ref_Mat, const int imgW, const int imgH, const int w, const int h, int &x, int &y){
    bool templateOkay = false;
    for(int j=int(imgH*0.15); j<int(imgH*0.85) - h; j+=imgH/40){
        for (int i=int(imgW*0.15); i<int(imgW*0.85) - w; i+=imgW/40){
            Rect r(i,j, w,h);
            // rectangle(img, r, Scalar(255, 0, 255), 2, LINE_AA);
            // imshow("template_rgb", img);
            // waitKey(0);
            Mat temp_Mat = ref_Mat(r);
            
            temp_Mat.convertTo(temp_Mat, CV_8UC1);
            set<uchar> sem_set;
            vector<uchar> sem_vec;
            for(int ii=0; ii<temp_Mat.rows; ii++){
                for(int jj=0; jj<temp_Mat.cols; jj++){
                    sem_set.insert(temp_Mat.at<uchar>(ii,jj));
                    sem_vec.push_back(temp_Mat.at<uchar>(ii,jj));
                }
            }
            
            //there must be at least 3 semantic labels in the template
            if(sem_set.size()<3){
                // cout<<"("<<i<<" ,"<<j<<")"<<sem_set.size()<<endl;
                continue;
            }
            //each type of the semantic label must occupy no more than 50% of the template 
            int counter=0;
            for(set<uchar>::iterator it = sem_set.begin(); it!=sem_set.end(); it++){
                int num = count(sem_vec.begin(), sem_vec.end(), *it);
                if (num > w*h*0.1){
                    counter+=1;
                }
            }
            if (counter<3){
                    templateOkay=false;
                    continue;
                }
            else{
                templateOkay=true;
                x=i;
                y=j;
            }
        }
        if(templateOkay==true){
            break;
        }
    }
    return templateOkay;
}

int main(int argc, char **argv)
{
    clock_t start, end; //timing
    double t_diff; //timing
    int imgW=1408; //img width
    int imgH=376;   //img height
    float MI;
    Point pt_max;
    double val_max, val_min;
    int imgSkip=0;

    //get histMat
    start=clock();
    FileStorage fs_read;
    Mat histMat;
    fs_read.open("histMat.xml", FileStorage::READ);
    if(!fs_read.isOpened()){
        cout<<"histMat.xml not opened"<<endl;
        exit(1);
    }
    fs_read["histMat"]>>histMat;
    fs_read.release();
    histMat = histMat.colRange(0,20).rowRange(0,20).clone();
    histMat.convertTo(histMat, CV_32FC1);
    end = clock(); //stop timing
    t_diff=(double)(end-start)/CLOCKS_PER_SEC; //calculate time difference
    printf("time to extract histMat %f \n", t_diff);


    vector<Mat> row_mat_vec(histMat.rows);
    for(int i=0; i<histMat.rows; i++){
        Mat &temperal = row_mat_vec[i];
        temperal = histMat.row(i) /sum(histMat.row(i))[0];
    }

    Mat row_mat[20][20];
    for(int i=0; i<20; i++){
        for(int j=0; j<20; j++){
            Mat P1 = row_mat_vec[i].t();
            Mat P2 = row_mat_vec[j];
            row_mat[i][j] = P1 * P2;
        }
    }

    




    //get all the picture names in segmentation folder and raw image folder
    // string seg_path=kitti360+"2013_05_28_drive_0007_sync_image_00/segmentation";
    // string seg_path=kitti360+"data_2d_raw/2013_05_28_drive_0010_sync/image_00/segmentation";
    string seg_path=kitti360+"data_2d_semantics/train/2013_05_28_drive_0010_sync/image_00/semantic";
    vector<cv::String> fn_seg;
    glob(seg_path, fn_seg, false);

    // string raw_path = kitti360+"data_2d_raw/2013_05_28_drive_0007_sync/image_00/data_rect";
    string raw_path = kitti360+"data_2d_raw/2013_05_28_drive_0010_sync/image_00/data_rect";
    vector<cv::String> fn_raw;
    glob(raw_path, fn_raw, false);

    // string segrgb_path = kitti360+"data_2d_raw/2013_05_28_drive_0010_sync/image_00/segmentation_rgb";
    string segrgb_path = kitti360+"data_2d_semantics/train/2013_05_28_drive_0010_sync/image_00/semantic_rgb";
    vector<cv::String> fn_segrgb;
    glob(segrgb_path, fn_segrgb, false);




    //perform template matching image by image
    // for(int i=0; i<fn_seg.size()-1; i++){
    for(int i=0; i<2680; i+=5){
        cout<<"current image"<<i<<endl;
        //read segmentation image and rgb image, select template and build result matrix
        Mat ref_Mat = imread(fn_seg[i], IMREAD_GRAYSCALE);
        Mat src_Mat = imread(fn_seg[i+1], IMREAD_GRAYSCALE);
        Mat ref_color = imread(fn_raw[i], IMREAD_COLOR);
        Mat src_color = imread(fn_raw[i+1], IMREAD_COLOR);
        Mat ref_segrgb = imread(fn_segrgb[i],IMREAD_COLOR);
        Mat src_segrgb = imread(fn_segrgb[i+1],IMREAD_COLOR);
        if (src_Mat.empty() || ref_Mat.empty() || ref_color.empty() || src_color.empty() 
            || ref_segrgb.empty() || src_segrgb.empty()){
            cout << "failed to read image" << endl;
            return EXIT_FAILURE;
        }

        // int x = 700;
        // int y = 160;
        // int w = 100;
        // int h = 80;
        // int x = 680;
        // int y = 160;
        // int w = 40;
        // int h = 60;
        // x= x-w/2;
        // y= y-h/2;
        // Rect r(int(imgW*5/8)+30,int(imgH/3)-20, 50,80); //801
        int w = 40;
        int h = 60;
        int x,y;
        bool templateOkay = templateFinder(ref_segrgb, ref_Mat, imgW, imgH, w, h, x, y);
        if(templateOkay==false){
            cout<<"image"<<to_string(i)<<"doesn't have a good template"<<endl;
            imgSkip+=1;
            cout<<"skipped img number: "<<imgSkip<<endl;
            continue;
        }
        
        Rect r(x,y, w,h);
        // Rect r(int(imgW/2)+80,int(imgH/2)-80, 10,20); //1901
        // Rect r(int(imgW/2)+100,int(imgH/2)-70, 10,20);
        // Rect r(int(imgW/2)+30,int(imgH/2)-20, 60,80); //101
        // Rect r(int(imgW/2)+30,int(imgH/2)-150, 60,80);
        // Rect r(int(imgW/2)+90,int(imgH/2)-70, 30,40);
        Mat temp_Mat = ref_Mat(r);
        Mat src(src_Mat.size(),CV_8UC1);
        src_Mat.convertTo(src, CV_8UC1);
        Mat temp(temp_Mat.size(),CV_8UC1);
        temp_Mat.convertTo(temp, CV_8UC1);
        Mat result = Mat::zeros(src.rows - temp.rows +1,src.cols - temp.cols +1, CV_32FC1);


        // // show template in rgb image
        // rectangle(ref_color, r, Scalar(255, 0, 255), 2, LINE_AA);
        // imshow("template_color", ref_color);
        // rectangle(ref_segrgb, r, Scalar(255, 0, 255), 2, LINE_AA);
        // imshow("template_rgb", ref_segrgb);
        // waitKey(0);
        // cout<<"program continues"<<endl;


        //build histogram for template image
        Mat temp_hist = Mat::ones(1,45, CV_32FC1)*(1e-6);
        Mat empty_mat;
        hist_builder(temp, empty_mat, temp_hist, row_mat_vec, row_mat);
        int N = temp.cols * temp.rows;
        cout<<"N: "<<N<<endl;

        

        //calculate I for each element of result matrix
        int counter;
        int interval=1000;
        
        for (int i=0; i<result.rows; i++)
        {
            counter=interval;
            
            start = clock();
            for (int j=0; j<result.cols; j++)
            {
                // cout<<"rrrr"<<endl;
                Mat src_part(src, cv::Rect(j,i,temp.cols, temp.rows));
                // cout<<"rrrr"<<endl;
                // Mat src_hist = Mat::ones(1,20, CV_32FC1)*(1e-6);
                // hist_builder(src_part, empty_mat, src_hist,row_mat_vec, row_mat);
                

                Mat temp_src_hist = Mat::ones(45,45, CV_32FC1)*(1e-6);
                // cout<<"rrrr"<<endl;
                hist_builder(temp, src_part, temp_src_hist,row_mat_vec, row_mat);
                // cout<<"rrrr"<<endl;
                Mat src_hist = Mat::zeros(1,45, CV_32FC1);
                for (int i=0; i<45; i++){
                    src_hist.at<float>(0,i) = sum(temp_src_hist.col(i))[0];
                }


                // cout<<sum(temp_src_hist.row(0))[0] - sum(temp_hist.col(0))[0]<<endl;
                // exit(0);


                MI = MI_calculator(temp_hist, src_hist, temp_src_hist, N);
                result.at<float>(i,j) = MI;
                if (j-counter==0){
                    end=clock();
                    t_diff=(double)(end-start)/CLOCKS_PER_SEC;
                    counter +=interval;
                    start=clock();
                    // cout<<"expected time left: "<<t_diff/interval * ((result.cols-j) + result.cols* (result.rows-i-1))<<endl;
                }
            }
        }
        

        //draw a rectangle on source image at the position where mutial information is the highest
        minMaxLoc(result, &val_min, &val_max, NULL, &pt_max);
        // cout<<val_max<<endl;
        // cout<<val_min<<endl;
        // cout<<pt_max.x<<" , "<<pt_max.y<<endl;
 
        rectangle(src_segrgb, Rect(pt_max.x, pt_max.y, temp.cols, temp.rows), Scalar(255, 0, 255), 2, LINE_AA);
        rectangle(src_color, Rect(pt_max.x, pt_max.y, temp.cols, temp.rows), Scalar(255, 0, 255), 2, LINE_AA);
        
        
        if(abs(x-pt_max.x)>30 || abs(y-pt_max.y)>30){
        // if(8>7){
            cout<<"match too far away: "<<i<<endl;
            cout<<abs(x-pt_max.x)<<", "<<abs(y-pt_max.y)<<endl;
            rectangle(ref_segrgb, r, Scalar(255, 0, 255), 2, LINE_AA);
            rectangle(src_segrgb, Rect(pt_max.x, pt_max.y, temp.cols, temp.rows), Scalar(255, 0, 255), 2, LINE_AA);
            float threshold = float((val_max-val_min)*0.99+val_min);
            for(int i=0; i<result.rows; i++){
                for(int j=0; j<result.cols; j++){
                    if(result.at<float>(i,j)>threshold){
                        // drawMarker(src_color, Point(j,i), Scalar(0,0,255), MARKER_TILTED_CROSS, 5, 1,8);
                        drawMarker(src_segrgb, Point(j,i), Scalar(255,255,255), MARKER_TILTED_CROSS, 5, 1,8);
                    }
                    }
            }
            // imshow("template_rgb", ref_segrgb);
            // imshow("src_rgb", src_segrgb);

            // //convert result to heatmap form
            Mat result_copy = result.clone();
            Mat heatMap;
            normalize(result_copy, heatMap, 0, 255, NORM_MINMAX);
            heatMap.convertTo(heatMap, CV_8UC1);
            applyColorMap(heatMap, heatMap, COLORMAP_JET);

            vector<int> compression_params;
            compression_params.push_back(IMWRITE_PNG_COMPRESSION);
            compression_params.push_back(3);
            bool flag = false;
            // flag = imwrite(kitti360 + "MI_images/strange_images/"+to_string(i)+"_color_temp.png", ref_color, compression_params);
            // flag = imwrite(kitti360 + "MI_images/strange_images/"+to_string(i)+"_color_match.png", src_color, compression_params);
            flag = imwrite(kitti360 + "MI_images/strange_images/"+to_string(i)+"_heatmap.png", heatMap, compression_params);
            flag = imwrite(kitti360 + "MI_images/strange_images/"+to_string(i)+"_seg_temp.png", ref_segrgb, compression_params);
            flag = imwrite(kitti360 + "MI_images/strange_images/"+to_string(i)+"_seg_match.png", src_segrgb, compression_params);
            // waitKey(0);
        }



        // //show histogram of best match
        // Mat src_part(src, cv::Rect(85,143,temp.cols, temp.rows));
        // Mat temp_src_hist = Mat::ones(20,20, CV_32FC1);
        // hist_builder(temp, src_part, temp_src_hist, histMat, row_sum);
        // cout<<temp_src_hist<<endl;
        
        
        // // scale the value in result matrix to 0 to 255
        // Mat norm_result;
        // normalize(result, norm_result, 0, 255, NORM_MINMAX);
        // norm_result.convertTo(norm_result,CV_8UC1);


        // //convert result to heatmap form
        // Mat result_copy = result.clone();
        // Mat heatMap;
        // // minMaxLoc(result_copy, &val_min, &val_max, NULL, NULL);
        // // float mid_val = float((val_max - val_min)/2 + val_min);
        // // // float mid_val = 0.;
        // // Mat mask0 = result_copy<mid_val;
        // // result_copy.setTo(0,mask0);
        // // Mat mask = result_copy>mid_val;
        // // normalize(result_copy, heatMap, 0, 255, NORM_MINMAX, -1,mask);
        // normalize(result_copy, heatMap, 0, 255, NORM_MINMAX);
        // heatMap.convertTo(heatMap, CV_8UC1);
        // applyColorMap(heatMap, heatMap, COLORMAP_JET);

        // float threshold = float((val_max-val_min)*0.95+val_min);
        // for(int i=0; i<result.rows; i++){
        //     for(int j=0; j<result.cols; j++){
        //         if(result.at<float>(i,j)>threshold){
        //             // rectangle(raw_image, Rect(j, i, 50, 80), Scalar(255, 0, 0), 1, LINE_AA);
        //             drawMarker(src_color, Point(j,i), Scalar(0,0,255), MARKER_TILTED_CROSS, 5, 1,8);
        //             drawMarker(src_segrgb, Point(j,i), Scalar(255,255,255), MARKER_TILTED_CROSS, 5, 1,8);
        //         }
        //         }
        // }


        // //save matrices to xml file
        // FileStorage fs_write("MIMat.xml", FileStorage::WRITE);
        // fs_write<<"result"<<result;
        // fs_write<<"norm_result"<<norm_result;
        // fs_write<<"temp"<<temp;
        // fs_write<<"heatMap"<<heatMap;
        // fs_write.release();


        // //display the images
        // // imshow("match",src_color);
        // // imshow("seg_match", src_segrgb);
        // // imshow("heatMap", heatMap);


        // //save images in CV_8U png form
        // vector<int> compression_params;
        // compression_params.push_back(IMWRITE_PNG_COMPRESSION);
        // compression_params.push_back(3);
        // bool flag = false;
        // // flag = imwrite(kitti360 + "MI_images/"+to_string(i)+"_color_temp.png", ref_color, compression_params);
        // // flag = imwrite(kitti360 + "MI_images/"+to_string(i)+"_color_match.png", src_color, compression_params);
        // flag = imwrite(kitti360 + "MI_images/"+to_string(i)+"_seg_temp.png", ref_segrgb, compression_params);
        // flag = imwrite(kitti360 + "MI_images/"+to_string(i)+"_seg_match.png", src_segrgb, compression_params);
        // flag = imwrite(kitti360 + "MI_images/"+to_string(i)+"_heatMap.png", heatMap, compression_params);

        // waitKey(0);
    }
}