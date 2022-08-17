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
                Pst = float(histtt.at<float>(i,j))/(N+380);
                Pt = float(temppp.at<float>(0,i))/N;
                Ps = float(srccc.at<float>(0,j))/N;
                I += Pst * log(Pst/(Ps*Pt));
            }
        }
        
        return I;

    }

// void Label_counter(const Mat input_mat, set<uchar>& label_set, vector<int>& label_count){
//     vector<uchar> input_mat_vec;
//     for(int i=0; i<input_mat.rows; i++){
//         for(int j=0; j<input_mat.cols; j++){
//             input_mat_vec.push_back(input_mat.at<uchar>(i,j));
//             label_set.insert(input_mat.at<uchar>(i,j));
//         }
//     }
//     for(set<uchar>::iterator it = label_set.begin(); it!=label_set.end(); it++){
//         label_count.push_back(count(input_mat_vec.begin(), input_mat_vec.end(), *it));
//     }
// }

void hist_builder(const Mat input_mat, const Mat input_mat2, Mat& output_mat, const Mat hist_mat, const vector<float> col_sum){
    // assert(input_mat.size==input_mat2.size);
    uchar P1;
    uchar P2;
    if(input_mat2.empty()){
        for(int i=0; i<input_mat.rows; ++i){
            for(int j=0; j<input_mat.cols; ++j){
                P1 = input_mat.at<uchar>(i,j);
                // output_mat.at<float>(0,P1) += 1;
                output_mat += hist_mat.col(P1).t() / col_sum[P1];
            }
        }
    }
    else{
        for(int i=0; i<input_mat.rows; ++i){
            for(int j=0; j<input_mat.cols; ++j){
                P1 = input_mat.at<uchar>(i,j);
                P2 = input_mat2.at<uchar>(i,j);
                Mat P1_mat = hist_mat.col(P1) / col_sum[P1];
                Mat P2_mat = hist_mat.col(P2).t() / col_sum[P2];

                cout<<int(P1)<<","<<P1_mat * P2_mat<<endl;
                exit(0);
                output_mat += P1_mat * P2_mat;
                
                // output_mat.at<float>(P1,P2) += 1;

            }
        }
    }

}

int main(int argc, char **argv)
{
    clock_t start, end; //timing
    double t_diff; //timing
    int tempWl = 200;   //template width
    int tempWr = 1200;
    int tempHu = 80;   //template hight
    int tempHd = 300;
    int imgW=1408; //img width
    int imgH=376;   //img height
    float MI;
    Point pt_max;
    double val_max, val_min;

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

    cout<<histMat<<endl;

    vector<float> col_sum;
    for(int i=0; i<histMat.cols; i++){
        col_sum.push_back(sum(histMat.col(i))[0]);
    }

    vector<Mat> col_mat_vec(col_sum.size());
    for(int i=0; i<histMat.cols; i++){
        Mat &temperal = col_mat_vec[i];
        temperal = histMat.col(i).t()/sum(histMat.col(i))[0];
    }




    //get all the picture names in segmentation folder and raw image folder
    string seg_path=kitti360+"2013_05_28_drive_0007_sync_image_00/segmentation";
    vector<cv::String> fn_seg;
    glob(seg_path, fn_seg, false);

    string raw_path = kitti360+"data_2d_raw/2013_05_28_drive_0007_sync/image_00/data_rect";
    vector<cv::String> fn_raw;
    glob(raw_path, fn_raw, false);


    //perform template matching image by image
    for(int i=1901; i<fn_seg.size()-1; i++){
        //read segmentation image and rgb image, select template and build result matrix
        Mat ref_Mat = imread(fn_seg[i], IMREAD_GRAYSCALE);
        Mat src_Mat = imread(fn_seg[i+1], IMREAD_GRAYSCALE);
        Mat ref_color = imread(fn_raw[i], IMREAD_COLOR);
        Mat src_color = imread(fn_raw[i+1], IMREAD_COLOR);
        if (src_Mat.empty() || ref_Mat.empty() || ref_color.empty() || src_color.empty())
        {
            cout << "failed to read image" << endl;
            return EXIT_FAILURE;
        }
        // Rect r(int(imgW*5/8)+30,int(imgH/3)-20, 50,80); //801
        // Rect r(int(imgW/2)+80,int(imgH/2)-80, 10,20); //1901
        Rect r(int(imgW/2)+100,int(imgH/2)-70, 10,20);
        // Rect r(int(imgW/2)+30,int(imgH/2)-20, 60,80); //101
        // Rect r(int(imgW/2)+30,int(imgH/2)-150, 60,80);
        // Rect r(int(imgW/2)+90,int(imgH/2)-70, 30,40);
        Mat temp_Mat = ref_Mat(r);
        Mat src(src_Mat.size(),CV_8UC1);
        src_Mat.convertTo(src, CV_8UC1);
        Mat temp(temp_Mat.size(),CV_8UC1);
        temp_Mat.convertTo(temp, CV_8UC1);
        Mat result = Mat::zeros(src.rows - temp.rows +1,src.cols - temp.cols +1, CV_32FC1);


        //show template in rgb image
        rectangle(ref_color, r, Scalar(255, 0, 255), 2, LINE_AA);
        imshow("template_color", ref_color);
        rectangle(ref_Mat, r, Scalar(255, 0, 255), 2, LINE_AA);
        imshow("template", ref_Mat);
        waitKey(0);
        cout<<"program continues"<<endl;


        //count the number of each semantic label in template
        // set<uchar> temp_label;
        // vector<int> temp_label_count;
        // Label_counter(temp, temp_label, temp_label_count);
        Mat temp_hist = Mat::ones(1,20, CV_32FC1);
        Mat empty_mat;
        hist_builder(temp, empty_mat, temp_hist, histMat, col_sum);
        int N = sum(temp_hist)[0];
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
                Mat src_part(src, cv::Rect(j,i,temp.cols, temp.rows));

                Mat src_hist = Mat::ones(1,20, CV_32FC1);
                hist_builder(src_part, empty_mat, src_hist, histMat, col_sum);

                Mat temp_src_hist = Mat::ones(20,20, CV_32FC1);
                hist_builder(temp, src_part, temp_src_hist, histMat, col_sum);


                MI = MI_calculator(temp_hist, src_hist, temp_src_hist, N);
                result.at<float>(i,j) = MI;
                if (j-counter==0){
                    end=clock();
                    t_diff=(double)(end-start)/CLOCKS_PER_SEC;
                    counter +=interval;
                    start=clock();
                    cout<<"expected time left: "<<t_diff/interval * ((result.cols-j) + result.cols* (result.rows-i-1))<<endl;
                }
            }
        }
        

        //draw a rectangle on source image at the position where mutial information is the highest
        minMaxLoc(result, &val_min, &val_max, NULL, &pt_max);
        cout<<val_max<<endl;
        cout<<val_min<<endl;
        cout<<pt_max.x<<" , "<<pt_max.y<<endl;
        Mat dst;
        src.copyTo(dst);
        rectangle(dst, Rect(pt_max.x, pt_max.y, temp.cols, temp.rows), Scalar(255, 0, 255), 2, LINE_AA);
        rectangle(src_color, Rect(pt_max.x, pt_max.y, temp.cols, temp.rows), Scalar(255, 0, 255), 2, LINE_AA);
        // scale the value in result matrix to 0 to 255
        Mat norm_result;
        normalize(result, norm_result, 0, 255, NORM_MINMAX);
        norm_result.convertTo(norm_result,CV_8UC1);


        //convert result to heatmap form
        Mat result_copy = result.clone();
        Mat heatMap;
        // minMaxLoc(result_copy, &val_min, &val_max, NULL, NULL);
        // float mid_val = float((val_max - val_min)/2 + val_min);
        // // float mid_val = 0.;
        // Mat mask0 = result_copy<mid_val;
        // result_copy.setTo(0,mask0);
        // Mat mask = result_copy>mid_val;
        // normalize(result_copy, heatMap, 0, 255, NORM_MINMAX, -1,mask);
        normalize(result_copy, heatMap, 0, 255, NORM_MINMAX);
        heatMap.convertTo(heatMap, CV_8UC1);
        applyColorMap(heatMap, heatMap, COLORMAP_JET);

        float threshold = float((val_max-val_min)*0.95+val_min);
        for(int i=0; i<result.rows; i++){
            for(int j=0; j<result.cols; j++){
                if(result.at<float>(i,j)>threshold){
                    // rectangle(raw_image, Rect(j, i, 50, 80), Scalar(255, 0, 0), 1, LINE_AA);
                    drawMarker(src_color, Point(j,i), Scalar(0,0,255), MARKER_TILTED_CROSS, 5, 1,8);
                }
                }
        }


        //save matrices to xml file
        FileStorage fs_write("MIMat.xml", FileStorage::WRITE);
        fs_write<<"result"<<result;
        fs_write<<"norm_result"<<norm_result;
        fs_write<<"dst"<<dst;
        fs_write<<"temp"<<temp;
        fs_write<<"heatMap"<<heatMap;
        fs_write.release();


        //display the images
        imshow("match",src_color);
        imshow("heatMap", heatMap);


        //save images in CV_8U png form
        vector<int> compression_params;
        compression_params.push_back(IMWRITE_PNG_COMPRESSION);
        compression_params.push_back(3);
        bool flag = false;
        flag = imwrite(kitti360 + "MI_images/color_temp_"+to_string(i)+".png", ref_color, compression_params);
        flag = imwrite(kitti360 + "MI_images/color_match_"+to_string(i)+".png", src_color, compression_params);
        // flag = imwrite(kitti360 + "MI_images/norm_result_"+to_string(i)+".png", norm_result, compression_params);
        // flag = imwrite(kitti360 + "MI_images/temp_"+to_string(i)+".png", temp, compression_params);
        // flag = imwrite(kitti360 + "MI_images/match_"+to_string(i)+".png", dst, compression_params);
        flag = imwrite(kitti360 + "MI_images/heatMap_"+to_string(i)+".png", heatMap, compression_params);

        // Mat re = imread(kitti360 + "MI_images/result.png", IMREAD_GRAYSCALE);
        // imshow("reread_result", re);
        
        // try
        // {
        //     flag = imwrite(kitti360 + "MI_images/result.png", result, compression_params);

        // }
        // catch (const cv::Exception& ex)
        // {
        //     fprintf(stderr, "Exception converting image to PNG format: %s\n", ex.what());
        // }
        // if (flag)
        //     printf("Saved PNG file with alpha data.\n");
        // else
        //     printf("ERROR: Can't save PNG file.\n");

        waitKey(0);
    }
}