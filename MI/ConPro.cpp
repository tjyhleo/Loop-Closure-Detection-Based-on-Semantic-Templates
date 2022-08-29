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


float ConPro_calculator(const Mat srcMat, const Mat tempMat, const Mat ConProMat)
    {
        float I=0.;
        //go through source image and template image, add conpro of each pair of corresponding pixel
        for (int i=0; i<srcMat.rows; i++)
            {
                for (int j=0; j<srcMat.cols; j++)
                { 
                    uchar P1 = tempMat.at<uchar>(i,j);
                    uchar P2 = srcMat.at<uchar>(i,j); 
                    float ConPro = ConProMat.at<float>(P1,P2);
                    I += ConPro; 
                }
            }     
        return I;
    }


vector<int> histCheck(Mat ConProMat){
    vector<int> ignoreLabels;
    for(int i = 0; i<ConProMat.rows; i++){
        float s = ConProMat.at<float>(i,i);
        if(s<0.7){
            ignoreLabels.push_back(i);
        }
    }
    return ignoreLabels;
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

bool templateFinder(Mat img, Mat ref_Mat, const int w, const int h, int &x, int &y, 
                const vector<int> ignoreLabels, const vector<int>goodLabels,const vector<int>badLabels ){
    bool templateOkay = false;
    bool lowerRequire=false;
    int imgH = ref_Mat.rows;
    int imgW = ref_Mat.cols;
    Mat goodTemp;
    Mat okayTemp;
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
                continue;
            }
            //there must be no bad semantic labels in template
            bool badLabelExist=false;
            for(set<uchar>::iterator it = sem_set.begin(); it!=sem_set.end(); it++){
                if(find(ignoreLabels.begin(), ignoreLabels.end(), *it) != ignoreLabels.end()){
                    badLabelExist=true;
                    break;
                };
            }
            if(badLabelExist==true){
                continue;
            }
            //there must be at least 3 semantic labels that occupy more than 20% of the template
            int counter=0;
            for(set<uchar>::iterator it = sem_set.begin(); it!=sem_set.end(); it++){
                int num = count(sem_vec.begin(), sem_vec.end(), *it);
                if (num > w*h*0.2){
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

void showSemNum(Mat temp){
    vector<uchar> template_labels_vec;
    set<uchar> template_labels_set;
    for(int i=0; i<temp.rows; i++){
        for(int j=0; j<temp.cols; j++){
            template_labels_vec.push_back(temp.at<uchar>(i,j));
            template_labels_set.insert(temp.at<uchar>(i,j));
        }
    }
    for(set<uchar>::iterator it = template_labels_set.begin(); it!=template_labels_set.end(); it++){
        cout<<int(*it)<<" : "<<count(template_labels_vec.begin(), template_labels_vec.end(), *it)<<endl;
    }
}


void ManualCheck(Mat tempMat, Mat srcMat, Mat ConProMat){
    Mat sem_count = Mat::zeros(80,80, CV_32SC1);
    float I=0.;
    for (int i=0; i<srcMat.rows; i++)
        {
            for (int j=0; j<srcMat.cols; j++)
            { 
                uchar P1 = tempMat.at<uchar>(i,j);
                uchar P2 = srcMat.at<uchar>(i,j); 
                float ConPro = ConProMat.at<float>(P1,P2);
                I+=ConPro;
                sem_count.at<int>(P1,P2) +=1;
            }
        } 
    cout<<"ConPro: "<<I<<endl;
    for(int i=0; i<sem_count.rows; i++){
        for(int j=0; j<sem_count.cols; j++){
            int count = sem_count.at<int>(i,j);
            if(count>0){
                cout<<i<<", "<<j<<": "<<count<<" * "<<ConProMat.at<float>(i,j)
                    << "  "<<int(count* ConProMat.at<float>(i,j) / I *100) <<"%"<<endl;
            }
        }
    }
    
}

vector<int> goodLabelFinder(){
    int tra_sign = 20;
    int tra_light = 19;
    vector<int> goodLabels;
    goodLabels.push_back(tra_sign);
    goodLabels.push_back(tra_light);
    return goodLabels;
}

vector<int> badLabelFinder(){
    int vegetation = 21;
    vector<int> badLabels;
    badLabels.push_back(vegetation);
    return badLabels;
}

int main(int argc, char **argv)
{
    clock_t start, end; //timing
    double t_diff; //timing
    float ConPro; //conditional probability
    Point pt_max, pt_min;
    double val_max, val_min;
    int imgSkip;
    vector<int> mismatch_vec;


    //get histMat
    Mat histMat;
    get_histMat(histMat);

    // cout<<histMat.diag()<<endl;

    //get sums of histMat
    vector<float> row_sum;
    for(int i=0; i<histMat.rows; i++){
        row_sum.push_back(sum(histMat.row(i))[0]);
    }

    vector<float> col_sum;
    for(int i=0; i<histMat.cols; i++){
        col_sum.push_back(sum(histMat.col(i))[0]);
    }

    float total = sum(histMat)[0];
    cout<<"histMat total: "<<total<<endl;

    Mat ConProMat = histMat.clone(); //conditional probability matrix
    ConProMat.convertTo(ConProMat,CV_32FC1);
    for(int i=0; i<ConProMat.rows; i++){
        ConProMat.row(i) = ConProMat.row(i) / row_sum[i];
    }
    
    vector<int> ignoreLabels = histCheck(ConProMat);
    vector<int> goodLabels = goodLabelFinder();
    vector<int> badLabels = badLabelFinder();
    



    //get all the picture names in segmentation folder and raw image folder
    string seg_path=kitti360+"data_2d_semantics/train/2013_05_28_drive_0010_sync/image_00/semantic";
    // string seg_path=kitti360+"data_2d_raw/2013_05_28_drive_0010_sync/image_00/segmentation";
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
    // for(int i=369; i<fn_seg.size()-1; i++){
    for(int i=125; i<2680; i+=5){
        int str_len = fn_seg[i].length();
        string frameId = fn_seg[i].substr(str_len-14, 10);
        cout<<"current image: "<<frameId<<endl;

        //read segmentation image and rgb image, select template and build result matrix
        Mat ref_Mat = imread(fn_seg[i], IMREAD_GRAYSCALE);
        Mat src_Mat = imread(fn_seg[i+1], IMREAD_GRAYSCALE);
        Mat ref_color = imread(fn_raw[i], IMREAD_COLOR);
        Mat src_color = imread(fn_raw[i+1], IMREAD_COLOR);
        Mat ref_segrgb = imread(fn_segrgb[i],IMREAD_COLOR);
        Mat src_segrgb = imread(fn_segrgb[i+1],IMREAD_COLOR);
        if (src_Mat.empty() || ref_Mat.empty() || ref_color.empty() || src_color.empty() 
            || ref_segrgb.empty() || src_segrgb.empty())
        {
            cout << "failed to read image" << endl;
            return EXIT_FAILURE;
        }

        int w = 30;
        int h = 30;
        int x,y;
        start = clock();
        bool templateOkay = templateFinder(ref_segrgb, ref_Mat, w, h, x, y, ignoreLabels, goodLabels, badLabels);
        end=clock();
        t_diff=(double)(end-start)/CLOCKS_PER_SEC;
        // cout<<"time to find template: "<<t_diff<<endl;
        if(templateOkay==false){
            cout<<"image"<<to_string(i)<<"doesn't have a good template"<<endl;
            imgSkip+=1;
            cout<<"skipped img number: "<<imgSkip<<endl;
            continue;
        }

        Rect r(x,y, w,h);
        Mat temp_Mat = ref_Mat(r);
        Mat src(src_Mat.size(),CV_8UC1);
        src_Mat.convertTo(src, CV_8UC1);
        Mat temp(temp_Mat.size(),CV_8UC1);
        temp_Mat.convertTo(temp, CV_8UC1);
        Mat result = Mat::zeros(src.rows - temp.rows +1,src.cols - temp.cols +1, CV_32FC1);


        //show template in rgb image
        // rectangle(ref_color, r, Scalar(255, 0, 255), 2, LINE_AA);
        // imshow("template_color", ref_color);
        // rectangle(ref_segrgb, r, Scalar(255, 0, 255), 2, LINE_AA);
        // imshow("template_rgb", ref_segrgb);
        // waitKey(0);
        // cout<<"program continues"<<endl;


        // //show the number of each semantic label in template
        // showSemNum(temp);


        //calculate ConPro for each element of result matrix
        int counter;
        int interval=1000;
        
        for (int i=0; i<result.rows; i++)
        {
            counter=interval;    
            start = clock();
            for (int j=0; j<result.cols; j++)
            {
                Mat src_part(src, cv::Rect(j,i,temp.cols, temp.rows));
                ConPro = ConPro_calculator(src_part, temp, ConProMat);
                result.at<float>(i,j) = ConPro;
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
        
        if(abs(x-pt_max.x)>50 || abs(y-pt_max.y)>50){
            mismatch_vec.push_back(i);
            cout<<"match too far away: "<<frameId<<endl;
            cout<<abs(x-pt_max.x)<<", "<<abs(y-pt_max.y)<<endl;
            rectangle(ref_segrgb, r, Scalar(255, 0, 255), 2, LINE_AA);
            rectangle(src_segrgb, Rect(pt_max.x, pt_max.y, temp.cols, temp.rows), Scalar(255, 0, 255), 2, LINE_AA);
            float threshold = float((val_max-val_min)*0.99+val_min);


            cout<<"ManualCheck for ground truth position"<<endl;
            Mat srcPart = src(r);
            ManualCheck(temp, srcPart, ConProMat);

            cout<<"ManualCheck for best match position"<<endl;
            srcPart = src(Rect(pt_max.x, pt_max.y, w,h));
            ManualCheck(temp, srcPart, ConProMat);
            // waitKey(0); 


            for(int i=0; i<result.rows; i++){
                for(int j=0; j<result.cols; j++){
                    if(result.at<float>(i,j)>threshold){
                        // drawMarker(src_color, Point(j,i), Scalar(0,0,255), MARKER_TILTED_CROSS, 5, 1,8);
                        drawMarker(src_segrgb, Point(j,i), Scalar(255,255,255), MARKER_TILTED_CROSS, 5, 1,8);
                    }
                    }
            }
            
            // minMaxLoc(result_copy, &val_min, &val_max, NULL, NULL);
            // float mid_val = float((val_max - val_min)/2 + val_min);
            // // float mid_val = 0.;
            // Mat mask0 = result_copy<mid_val;
            // result_copy.setTo(0,mask0);
            // Mat mask = result_copy>mid_val;
            // normalize(result_copy, heatMap, 0, 255, NORM_MINMAX, -1,mask);
            //transfer result into heatmap
            Mat heatMap;
            normalize(result, heatMap, 0, 255, NORM_MINMAX);
            heatMap.convertTo(heatMap, CV_8UC1);
            applyColorMap(heatMap, heatMap, COLORMAP_JET);


            vector<int> compression_params;
            compression_params.push_back(IMWRITE_PNG_COMPRESSION);
            compression_params.push_back(3);
            bool flag = false;
            // flag = imwrite(kitti360 + "MI_images/"+frameId+"_color_temp.png", ref_color, compression_params);
            // flag = imwrite(kitti360 + "MI_images/"+frameId+"_color_match.png", src_color, compression_params);
            flag = imwrite(kitti360 + "MI_images/strange_images/"+frameId+"_seg_temp.png", ref_segrgb, compression_params);
            flag = imwrite(kitti360 + "MI_images/strange_images/"+frameId+"_seg_match.png", src_segrgb, compression_params);
            flag = imwrite(kitti360 + "MI_images/strange_images/"+frameId+"_heatMap.png", heatMap, compression_params);
            // imshow("template_rgb", ref_segrgb);
            // imshow("src_rgb", src_segrgb);
            // waitKey(0);
        }
    }
    for(int i=0; i<mismatch_vec.size(); i++){
            cout<<i<<endl;
    }
}