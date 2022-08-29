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
float MI_calculator(const Mat srccc, const Mat temppp, const Mat histtt, const vector<float> row_sum, const vector<float> col_sum, float total, vector<float> &Pst_list)
    {
        clock_t start, end; //timing
        double t_diff; //timing
        float I=0.;

        //go through source image and template image, add the value of each pixel to corresponding position in matrix his
        for (int i=0; i<temppp.rows; i++)
            {
                for (int j=0; j<temppp.cols; j++)
                {
                    float Ps, Pt, Pst, subtotal; 
                    uchar P1 = temppp.at<uchar>(i,j);
                    uchar P2 = srccc.at<uchar>(i,j); 
                    Pst = float(histtt.at<int>(P1,P2));
                    Pt = row_sum[P1];
                    Ps = col_sum[P2];
                    subtotal = Pt;
                    // subtotal = Ps + Pt - Pst;
            
                    // if(Pst/subtotal>0.1 && count(Pst_list.begin(), Pst_list.end(), Pst)==0){
                    //     cout<<int(P1)<<" , "<<int(P2)<<": "<<Pst<<": "<<Pst/subtotal<<endl;
                    //     Pst_list.push_back(Pst);
                    // }

                    // if(P1==14 && P2==4){
                    //     cout<<"14,2 "<<Pst/subtotal<<endl;
                    //     exit(0);
                    // }
                    
                    I += Pst/subtotal;    
                }
            }
        return I;
    }

vector<int> histCheck(Mat histMat){
    vector<int> badRows;
    for(int i = 0; i<histMat.rows; i++){
        for (int j=0; j<histMat.cols; j++){
            if(j!=i && histMat.at<int>(i,j)>histMat.at<int>(i,i)*0.5){
                badRows.push_back(i);
                // printf("%d",i);
                // cout<<histMat.row(i)<<endl;
                break;
            }
        }
    }
    return badRows;
    
}


bool templateFinder(Mat ref_Mat, const int imgW, const int imgH, const int w, const int h, int &x, int &y, const vector<int> badRows ){
    bool templateOkay = false;
    for(int j=int(imgH*0.15); j<int(imgH*0.85) - h; j+=imgH/30){
        for (int i=int(imgW*0.15); i<int(imgW*0.85) - w; i+=imgW/30){
            Rect r(i,j, w,h);
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
            //and if there is a bad semantic label (in badRows), it cannot occupy more than 5% of the template 
            for(set<uchar>::iterator it = sem_set.begin(); it!=sem_set.end(); it++){
                int num = count(sem_vec.begin(), sem_vec.end(), *it);
                if (num > w*h*0.5){
                    templateOkay=false;
                    break;
                }
                if(find(badRows.begin(), badRows.end(), *it) != badRows.end() && num>w*h*0.05){
                    templateOkay=false;
                    break;
                }
                else{
                    templateOkay=true;
                    x=i;
                    y=j;
                }
            }

            if(templateOkay==false){
                continue;
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
    int imgSkip = 0;
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
    end = clock(); //stop timing
    t_diff=(double)(end-start)/CLOCKS_PER_SEC; //calculate time difference
    printf("time to extract histMat %f \n", t_diff);
    minMaxLoc(histMat, &val_max, &val_min, NULL, NULL);
    cout<<"histMat min: "<<val_max<<endl;
    cout<<"histMat_max: "<<val_min<<endl;
    histMat = histMat.colRange(0,20).rowRange(0,20).clone();
    vector<int> badRows = histCheck(histMat);


    //get sums of rows and columns of histMat
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
    

    vector<float> Pst_list;


    //get all the picture names in segmentation folder and raw image folder
    // string seg_path=kitti360+"2013_05_28_drive_0007_sync_image_00/segmentation";
    string seg_path=kitti360+"data_2d_raw/2013_05_28_drive_0010_sync/image_00/segmentation";
    vector<cv::String> fn_seg;
    glob(seg_path, fn_seg, false);

    // string raw_path = kitti360+"data_2d_raw/2013_05_28_drive_0007_sync/image_00/data_rect";
    string raw_path = kitti360+"data_2d_raw/2013_05_28_drive_0010_sync/image_00/data_rect";
    vector<cv::String> fn_raw;
    glob(raw_path, fn_raw, false);

    string segrgb_path = kitti360+"data_2d_raw/2013_05_28_drive_0010_sync/image_00/segmentation_rgb";
    vector<cv::String> fn_segrgb;
    glob(segrgb_path, fn_segrgb, false);


    //perform template matching image by image
    // for(int i=369; i<fn_seg.size()-1; i++){
    for(int i=0; i<3800; i+=5){
        cout<<"current image"<<i<<endl;
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

        
        int w = 40;
        int h = 60;
        int x,y;
        bool templateOkay = templateFinder(ref_Mat, imgW, imgH, w, h, x, y, badRows);

        if(templateOkay==false){
            cout<<"image"<<to_string(i)<<"doesn't have a good template"<<endl;
            imgSkip+=1;
            cout<<"skipped img number: "<<imgSkip<<endl;
            continue;
        }
        
    
        // Rect r(int(imgW*5/8)+30,int(imgH/3)-20, 50,80); //801
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

        // Mat diagMat = histMat.colRange(0,20).rowRange(0,20).diag();
        // cout<<diagMat<<endl;


        // //show the number of each semantic label in template
        // vector<uchar> template_labels_vec;
        // set<uchar> template_labels_set;
        // for(int i=0; i<temp.rows; i++){
        //     for(int j=0; j<temp.cols; j++){
        //         template_labels_vec.push_back(temp.at<uchar>(i,j));
        //         template_labels_set.insert(temp.at<uchar>(i,j));
        //     }
        // }
        // for(set<uchar>::iterator it = template_labels_set.begin(); it!=template_labels_set.end(); it++){
        //     cout<<int(*it)<<" : "<<count(template_labels_vec.begin(), template_labels_vec.end(), *it)<<endl;
        // }


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
                MI = MI_calculator(src_part, temp, histMat, row_sum, col_sum, total, Pst_list);
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
        
        if(abs(x-pt_max.x)>150 || abs(y-pt_max.y)>30){
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

            Mat result_copy = result.clone();
            Mat heatMap;
            normalize(result_copy, heatMap, 0, 255, NORM_MINMAX);
            heatMap.convertTo(heatMap, CV_8UC1);
            applyColorMap(heatMap, heatMap, COLORMAP_JET);

            vector<int> compression_params;
            compression_params.push_back(IMWRITE_PNG_COMPRESSION);
            compression_params.push_back(3);
            bool flag = false;
            // flag = imwrite(kitti360 + "MI_images/"+to_string(i)+"_color_temp.png", ref_color, compression_params);
            // flag = imwrite(kitti360 + "MI_images/"+to_string(i)+"_color_match.png", src_color, compression_params);
            flag = imwrite(kitti360 + "MI_images/strange_images/"+to_string(i)+"_heatmap.png", heatMap, compression_params);
            flag = imwrite(kitti360 + "MI_images/strange_images/"+to_string(i)+"_seg_temp.png", ref_segrgb, compression_params);
            flag = imwrite(kitti360 + "MI_images/strange_images/"+to_string(i)+"_seg_match.png", src_segrgb, compression_params);
            // waitKey(0);
        }



        // rectangle(src_segrgb, Rect(pt_max.x, pt_max.y, temp.cols, temp.rows), Scalar(255, 0, 255), 2, LINE_AA);
        // rectangle(src_color, Rect(pt_max.x, pt_max.y, temp.cols, temp.rows), Scalar(255, 0, 255), 2, LINE_AA);
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


        // // //save matrices to xml file
        // // FileStorage fs_write("MIMat.xml", FileStorage::WRITE);
        // // fs_write<<"result"<<result;
        // // fs_write<<"norm_result"<<norm_result;
        // // fs_write<<"temp"<<temp;
        // // fs_write<<"heatMap"<<heatMap;
        // // fs_write.release();


        // //display the images
        // imshow("match",src_color);
        // imshow("seg_match", src_segrgb);
        // imshow("heatMap", heatMap);


        // //save images in CV_8U png form
        // vector<int> compression_params;
        // compression_params.push_back(IMWRITE_PNG_COMPRESSION);
        // compression_params.push_back(3);
        // bool flag = false;
        // flag = imwrite(kitti360 + "MI_images/"+to_string(i)+"_color_temp.png", ref_color, compression_params);
        // flag = imwrite(kitti360 + "MI_images/"+to_string(i)+"_color_match.png", src_color, compression_params);
        // flag = imwrite(kitti360 + "MI_images/"+to_string(i)+"_seg_temp.png", ref_segrgb, compression_params);
        // flag = imwrite(kitti360 + "MI_images/"+to_string(i)+"_seg_match.png", src_segrgb, compression_params);
        // flag = imwrite(kitti360 + "MI_images/"+to_string(i)+"_heatMap.png", heatMap, compression_params);

        
        // // try
        // // {
        // //     flag = imwrite(kitti360 + "MI_images/result.png", result, compression_params);

        // // }
        // // catch (const cv::Exception& ex)
        // // {
        // //     fprintf(stderr, "Exception converting image to PNG format: %s\n", ex.what());
        // // }
        // // if (flag)
        // //     printf("Saved PNG file with alpha data.\n");
        // // else
        // //     printf("ERROR: Can't save PNG file.\n");

        // waitKey(0);
    }
}