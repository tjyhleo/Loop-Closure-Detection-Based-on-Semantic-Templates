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

//function to calculate mutual information :I 
//  src: part of source image that overlapps with template image
//  templ: template image
//  templ_class: the set that contains the unique values in template image
float MI_calculator(const Mat srccc, const Mat temppp, const Mat histtt)
    {
        clock_t start, end; //timing
        double t_diff; //timing
        float I=0.;
        // float total = sum(histtt)[0];
        // cout<<"total: "<<total<<endl;


        // I=0.;

        //go through source image and template image, add the value of each pixel to corresponding position in matrix his
        for (int i=0; i<temppp.rows; i++)
            {
                for (int j=0; j<temppp.cols; j++)
                {
                    // start = clock();
                    //Ps: probability distribution of reference image
                    //Pt: probability distribution of template image
                    //Pst: joint probability distribution
                    float Ps, Pt, Pst, total; 
                    uchar P1 = temppp.at<uchar>(i,j); //temppp: template image matrix
                    uchar P2 = srccc.at<uchar>(i,j); //srccc: source image matrix
                    Pst = float(histtt.at<int>(P1,P2)); //histtt: hitogram matrix
                    Pt = sum(histtt.row(P1))[0];
                    Ps = sum(histtt.col(P2))[0];
                    total = Pt + Ps - Pst;
                    if((Pst * total) < (Ps*Pt)){
                        cout<<"Pst: "<<Pst<<endl;
                        cout<<"total: "<<total<<endl;
                        cout<<"Ps: "<<Ps<<endl;
                        cout<<"Pt: "<<Pt<<endl;
                        exit(1);
                    }
                    // total = Pt + Ps - Pst;
                    I += Pst/total * log(Pst*total/(Ps*Pt)); //total:sum of histogram matrix elements
                    // end = clock(); //stop timing
                    // t_diff=(double)(end-start)/CLOCKS_PER_SEC; //calculate time difference
                    // printf("one loop  %f \n", t_diff);
                }
            }
        
        return I;

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


    ////////////////////////////////////////////////////////////
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
    cout<<val_max<<endl;
    cout<<val_min<<endl;
    int total = sum(histMat)[0];
    cout<<total<<endl;

    // cout<<histMat.colRange(0,30).rowRange(0,30)<<endl;
    //get all the picture names in segmentation folder
    string seg_path=kitti360+"2013_05_28_drive_0007_sync_image_00/segmentation";
    vector<cv::String> fn_seg;
    glob(seg_path, fn_seg, false);


    // for(int i=0; i<fn_seg.size()-1; i++){
    for(int i=801; i<802; i++){
        
        Mat ref_Mat = imread(fn_seg[i], IMREAD_GRAYSCALE);
        Mat src_Mat = imread(fn_seg[i+2], IMREAD_GRAYSCALE);
        if (src_Mat.empty() || ref_Mat.empty())
        {
            cout << "failed to read image" << endl;
            return EXIT_FAILURE;
        }
        Rect r(int(imgW*5/8)+30,int(imgH/3)-20, 3,5);
        Mat temp_Mat = ref_Mat(r);
        Mat src(src_Mat.size(),CV_8UC1);
        src_Mat.convertTo(src, CV_8UC1);
        Mat temp(temp_Mat.size(),CV_8UC1);
        temp_Mat.convertTo(temp, CV_8UC1);
        Mat result = Mat::zeros(src.rows - temp.rows +1,src.cols - temp.cols +1, CV_32FC1);
        // cout<<result.size<<endl;

        imshow("template", temp);
        waitKey(0);

        cout<<"program continues"<<endl;


        //calculate I for each element of result
        int counter;
        int interval=1000;
        
        for (int i=0; i<result.rows; i++)
        {
            counter=interval;
            
            start = clock();
            for (int j=0; j<result.cols; j++)
            {
                // start = clock();
                Mat src_part(src, cv::Rect(j,i,temp.cols, temp.rows));
                MI = MI_calculator(src_part, temp, histMat);
                result.at<float>(i,j) = MI;
                // end = clock(); //stop timing
                // t_diff=(double)(end-start)/CLOCKS_PER_SEC; //calculate time difference
                // printf("time to calculate one MI value %f \n", t_diff);
                if (j-counter==0){
                    // cout<<ff%100<<endl;
                    end=clock();
                    t_diff=(double)(end-start)/CLOCKS_PER_SEC;
                    counter +=interval;
                    start=clock();
                    cout<<"expected time left: "<<t_diff/interval * ((result.cols-j) + result.cols* (result.rows-i-1))<<endl;
                }

            }
        }
        
        
        // cout <<"result: "<<result<<endl;

        //draw a rectangle on source image at the position where mutial information is the highest
        minMaxLoc(result, &val_max, &val_min, NULL, &pt_max);
        cout<<val_max<<endl;
        cout<<val_min<<endl;
        cout<<pt_max.x<<" , "<<pt_max.y<<endl;
        Mat dst;
        src.copyTo(dst);
        rectangle(dst, Rect(pt_max.x, pt_max.y, temp.cols, temp.rows), Scalar(255, 0, 255), 2, LINE_AA);

        // scale the value in result matrix to 0 to 255
        Mat norm_result;
        normalize(result, norm_result, 0, 255, NORM_MINMAX);

        //convert result to heatmap form
        Mat result_copy = result.clone();
        Mat heatMap;
        minMaxLoc(result_copy, &val_min, &val_max, NULL, NULL);
        float mid_val = float((val_max - val_min)*2/3 + val_min);
        Mat mask0 = result_copy<mid_val;
        result_copy.setTo(0,mask0);
        Mat mask = result_copy>mid_val;
        normalize(result_copy, heatMap, 0, 255, NORM_MINMAX, -1,mask);
        heatMap.convertTo(heatMap, CV_8UC1);
        applyColorMap(heatMap, heatMap, COLORMAP_JET);


        //save matrices to xml file
        FileStorage fs_write("MIMat.xml", FileStorage::WRITE);
        fs_write<<"result"<<result;
        fs_write<<"norm_result"<<norm_result;
        fs_write<<"dst"<<dst;
        fs_write<<"temp"<<temp;
        fs_write<<"heatMap"<<heatMap;
        fs_write.release();
        



        //display the images
        imshow("temp", temp);
        imshow("match",dst);
        imshow("norm_result",norm_result);
        imshow("heatMap", heatMap);


        //save images in CV_8U png form
        vector<int> compression_params;
        compression_params.push_back(IMWRITE_PNG_COMPRESSION);
        compression_params.push_back(3);

        bool flag = false;
        flag = imwrite(kitti360 + "MI_images/result.png", norm_result, compression_params);
        flag = imwrite(kitti360 + "MI_images/temp.png", temp, compression_params);
        flag = imwrite(kitti360 + "MI_images/match.png", dst, compression_params);
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