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

void entropy_calculator(Mat hist, float &Ht, float &Hs, float &Hts){
    // float Ht=0., Hs=0., Hts=0.;
    Mat temporal = hist.clone();
    // float N = sum(temporal)[0];
    temporal = temporal / 2400;
    for(int i=0; i<temporal.rows; i++){
        for(int j=0; j<temporal.cols; j++){
            Hts -= temporal.at<float>(i,j) * log(temporal.at<float>(i,j));
            Ht -= temporal.at<float>(i,j) * log(sum(temporal.row(i))[0]);
            Hs -= temporal.at<float>(i,j) * log(sum(temporal.col(j))[0]);
        }
    }
    // return Ht, Hs, Hts;
}


float MI_calculator(const Mat temppp, const Mat srccc, const Mat histtt, const int N)
    {
        clock_t start, end; //timing
        double t_diff; //timing
        float I=0.;
        float Pst, Pt, Ps;
        for(int i=0; i<histtt.rows; ++i){
            for(int j=0; j<histtt.cols; ++j){
                Pst = histtt.at<float>(i,j)/N;
                Pt = temppp.at<float>(0,i)/N;
                Ps = srccc.at<float>(0,j)/N;
                
                I += Pst * log(Pst/(Ps*Pt));
            }
        }

        // float Pst =0., Pt=0., Ps=0.;
        // entropy_calculator(histtt, Pt, Ps, Pst);
        // I = Pt + Ps - Pst;
        
        return I;

    }


void hist_builder(const Mat input_mat, const Mat input_mat2, Mat& output_mat, 
            const Mat hist_mat, const vector<Mat>row_mat_vec, const Mat row_mat[50][50]){
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
                // output_mat += row_mat[P1][P2];
                output_mat.at<float>(P1,P2) += 1;

            }
        }
    }

}

void get_histMat(Mat &histMat){
    FileStorage fs_read;
    double val_min, val_max;
    fs_read.open("histMat.xml", FileStorage::READ);
    if(!fs_read.isOpened()){
        cout<<"histMat.xml not opened"<<endl;
        exit(1);
    }
    fs_read["histMat"]>>histMat;
    fs_read.release();
    histMat = histMat.rowRange(0,50).colRange(0,50).clone();
    minMaxLoc(histMat, &val_min, &val_max, NULL, NULL);
    cout<<"histMat min, max: "<<val_min<<", "<<val_max<<endl;
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

    //get histMat
    Mat histMat;
    get_histMat(histMat);

 
    vector<Mat> row_mat_vec(histMat.rows);
    for(int i=0; i<histMat.rows; i++){
        Mat &temperal = row_mat_vec[i];
        temperal = histMat.row(i) /sum(histMat.row(i))[0];
    }

    Mat row_mat[50][50];
    for(int i=0; i<50; i++){
        for(int j=0; j<50; j++){
            Mat P1 = row_mat_vec[i].t();
            Mat P2 = row_mat_vec[j];
            row_mat[i][j] = P1 * P2;
        }
    }


    //get all the picture names in segmentation folder and raw image folder
    // string seg_path=kitti360+"data_2d_raw/2013_05_28_drive_0010_sync/image_00/segmentation";
    string seg_path=kitti360+"data_2d_semantics/train/2013_05_28_drive_0010_sync/image_00/semantic";
    vector<cv::String> fn_seg;
    glob(seg_path, fn_seg, false);

    string raw_path = kitti360+"data_2d_raw/2013_05_28_drive_0010_sync/image_00/data_rect";
    vector<cv::String> fn_raw;
    glob(raw_path, fn_raw, false);

    // string segrgb_path = kitti360+"data_2d_raw/2013_05_28_drive_0010_sync/image_00/segmentation_rgb";
    string segrgb_path=kitti360+"data_2d_semantics/train/2013_05_28_drive_0010_sync/image_00/semantic_rgb";
    vector<cv::String> fn_segrgb;
    glob(segrgb_path, fn_segrgb, false);




    //perform template matching image by image
    // for(int i=0; i<fn_seg.size()-1; i++){
    for(int i=0; i<500; i+=10){
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
        // Rect r(int(imgW*5/8)+30,int(imgH/3)-20, 50,80); //801
        // int x = 680;
        int x = 730;
        int y = 160;
        int w = 40;
        int h = 60;
        x= x-w/2;
        y= y-h/2;
        // Rect r(int(imgW*5/8)+30,int(imgH/3)-20, 50,80); //801
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


        //show template in rgb image
        // rectangle(ref_color, r, Scalar(255, 0, 255), 2, LINE_AA);
        // imshow("template_color", ref_color);
        rectangle(ref_segrgb, r, Scalar(255, 0, 255), 2, LINE_AA);
        // imshow("template_rgb", ref_segrgb);
        // waitKey(0);
        // cout<<"program continues"<<endl;


        //build histogram for template image
        Mat temp_hist = Mat::ones(1,20, CV_32FC1)*(1e-12);
        Mat empty_mat;
        hist_builder(temp, empty_mat, temp_hist, histMat, row_mat_vec, row_mat);
        cout<<"temp_hist: "<<temp_hist<<endl;
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
                Mat src_part(src, cv::Rect(j,i,temp.cols, temp.rows));

                // Mat src_hist = Mat::ones(1,20, CV_32FC1)*(1e-12);
                // hist_builder(src_part, empty_mat, src_hist, histMat, row_mat_vec, row_mat);
                

                Mat temp_src_hist = Mat::ones(20,20, CV_32FC1)*(1e-12);
                hist_builder(temp, src_part, temp_src_hist, histMat, row_mat_vec, row_mat);

                Mat src_hist = Mat::zeros(1,20, CV_32FC1);
                for (int i=0; i<20; i++){
                    src_hist.at<float>(0,i) = sum(temp_src_hist.col(i))[0];
                }

                Mat temp_hist = Mat::zeros(1,20, CV_32FC1);
                for (int i=0; i<20; i++){
                    temp_hist.at<float>(0,i) = sum(temp_src_hist.row(i))[0];
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
        cout<<val_max<<endl;
        cout<<val_min<<endl;
        cout<<pt_max.x<<" , "<<pt_max.y<<endl;
 
        rectangle(src_segrgb, Rect(pt_max.x, pt_max.y, temp.cols, temp.rows), Scalar(255, 0, 255), 2, LINE_AA);
        rectangle(src_color, Rect(pt_max.x, pt_max.y, temp.cols, temp.rows), Scalar(255, 0, 255), 2, LINE_AA);
        
        
        // if(abs(x-pt_max.x)>30 || abs(y-pt_max.y)>30){
        // // if(8>7){
        //     cout<<"match too far away: "<<i<<endl;
        //     cout<<abs(x-pt_max.x)<<", "<<abs(y-pt_max.y)<<endl;
        //     rectangle(ref_segrgb, r, Scalar(255, 0, 255), 2, LINE_AA);
        //     rectangle(src_segrgb, Rect(pt_max.x, pt_max.y, temp.cols, temp.rows), Scalar(255, 0, 255), 2, LINE_AA);
        //     float threshold = float((val_max-val_min)*0.99+val_min);
        //     for(int i=0; i<result.rows; i++){
        //         for(int j=0; j<result.cols; j++){
        //             if(result.at<float>(i,j)>threshold){
        //                 // drawMarker(src_color, Point(j,i), Scalar(0,0,255), MARKER_TILTED_CROSS, 5, 1,8);
        //                 drawMarker(src_segrgb, Point(j,i), Scalar(255,255,255), MARKER_TILTED_CROSS, 5, 1,8);
        //             }
        //             }
        //     }
        //     // imshow("template_rgb", ref_segrgb);
        //     // imshow("src_rgb", src_segrgb);

        //     vector<int> compression_params;
        //     compression_params.push_back(IMWRITE_PNG_COMPRESSION);
        //     compression_params.push_back(3);
        //     bool flag = false;
        //     // flag = imwrite(kitti360 + "MI_images/"+to_string(i)+"_color_temp.png", ref_color, compression_params);
        //     // flag = imwrite(kitti360 + "MI_images/"+to_string(i)+"_color_match.png", src_color, compression_params);
        //     flag = imwrite(kitti360 + "MI_images/strange_images/"+to_string(i)+"_seg_temp.png", ref_segrgb, compression_params);
        //     flag = imwrite(kitti360 + "MI_images/strange_images/"+to_string(i)+"_seg_match.png", src_segrgb, compression_params);
        //     // waitKey(0);
        // }



        //show histogram of best match
        Mat src_part(src, cv::Rect(x+20,y+20,temp.cols, temp.rows));
        Mat src_hist = Mat::ones(1,20, CV_32FC1)*(1e-12);
        hist_builder(src_part, empty_mat, src_hist, histMat, row_mat_vec, row_mat);
        cout<<"src_hist: "<<src_hist<<endl;
        Mat temp_src_hist = Mat::ones(20,20, CV_32FC1)*(1e-12);
        hist_builder(temp, src_part, temp_src_hist, histMat, row_mat_vec, row_mat);
        cout<<"temp_src_hist: "<<temp_src_hist<<endl;

        float Ht=0., Hs=0., Hts=0.;
        entropy_calculator(temp_src_hist, Ht, Hs, Hts);
        cout<<"template entropy: "<<Ht<<endl;
        cout<<"source entropy: "<<Hs<<endl;
        cout<<"joint entropy: "<<Hts<<endl;
        cout<<"MI by entropy: "<<Ht+Hs-Hts<<endl;

        float I = MI_calculator(temp_hist, src_hist, temp_src_hist, N);
        cout<<"MI: "<<I<<endl;
        
        
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
                    drawMarker(src_segrgb, Point(j,i), Scalar(255,255,255), MARKER_TILTED_CROSS, 5, 1,8);
                }
                }
        }


        //save matrices to xml file
        FileStorage fs_write("MIMat.xml", FileStorage::WRITE);
        fs_write<<"result"<<result;
        fs_write<<"norm_result"<<norm_result;
        fs_write<<"temp"<<temp;
        fs_write<<"heatMap"<<heatMap;
        fs_write.release();


        //display the images
        // imshow("match",src_color);
        // imshow("seg_match", src_segrgb);
        // imshow("heatMap", heatMap);


        //save images in CV_8U png form
        vector<int> compression_params;
        compression_params.push_back(IMWRITE_PNG_COMPRESSION);
        compression_params.push_back(3);
        bool flag = false;
        // flag = imwrite(kitti360 + "MI_images/"+to_string(i)+"_color_temp.png", ref_color, compression_params);
        // flag = imwrite(kitti360 + "MI_images/"+to_string(i)+"_color_match.png", src_color, compression_params);
        flag = imwrite(kitti360 + "MI_images/"+to_string(i)+"_seg_temp.png", ref_segrgb, compression_params);
        flag = imwrite(kitti360 + "MI_images/"+to_string(i)+"_seg_match.png", src_segrgb, compression_params);
        flag = imwrite(kitti360 + "MI_images/"+to_string(i)+"_heatMap.png", heatMap, compression_params);

        waitKey(0);
    }
}