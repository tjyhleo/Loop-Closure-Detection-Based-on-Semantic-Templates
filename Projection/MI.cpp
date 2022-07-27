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
        // start = clock();
        int total = sum(histtt)[0];
        // end = clock(); //stop timing
        // t_diff=(double)(end-start)/CLOCKS_PER_SEC; //calculate time difference
        // printf("sum of total %f \n", t_diff);

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
                    float Ps, Pt, Pst; 
                    uchar P1 = temppp.at<uchar>(i,j); //temppp: template image matrix
                    uchar P2 = srccc.at<uchar>(i,j); //srccc: source image matrix
                    Pst = float(histtt.at<int>(P1,P2)); //histtt: hitogram matrix
                    Pt = sum(histtt.row(P1))[0]/total;
                    Ps = sum(histtt.col(P2))[0]/total;
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
    float MI;

    //get histMat
    start=clock();
    FileStorage fs;
    Mat histMat;
    fs.open("histMat.xml", FileStorage::READ);
    if(!fs.isOpened()){
        cout<<"histMat.xml not opened"<<endl;
        exit(1);
    }
    fs["histMat"]>>histMat;
    fs.release();
    end = clock(); //stop timing
    t_diff=(double)(end-start)/CLOCKS_PER_SEC; //calculate time difference
    printf("time to extract histMat %f \n", t_diff);

    cout<<histMat.colRange(0,30).rowRange(0,30)<<endl;
    //get all the picture names in segmentation folder
    string seg_path=kitti360+"2013_05_28_drive_0007_sync_image_00/segmentation";
    vector<cv::String> fn_seg;
    glob(seg_path, fn_seg, false);
    // for(int i=0; i<fn_seg.size()-1; i++){
    for(int i=2690; i<2700; i++){
        
        Mat ref_Mat = imread(fn_seg[i], IMREAD_GRAYSCALE);
        Mat src_Mat = imread(fn_seg[i+1], IMREAD_GRAYSCALE);
        if (src_Mat.empty() || ref_Mat.empty())
        {
            cout << "failed to read image" << endl;
            return EXIT_FAILURE;
        }
        Rect r(tempWl,tempHu, 5,10);
        // Rect r(200,80,1000,220);
        // cout<<r<<endl;
        // cout<<ref_Mat.size<<endl;
        // cout<<"115 reached"<<endl;
        Mat temp_Mat = ref_Mat(r);
        // cout<<"116 reached"<<endl;
        //convert data type to CV_32FC1
        Mat src(src_Mat.size(),CV_8UC1);
        src_Mat.convertTo(src, CV_8UC1);
        Mat temp(temp_Mat.size(),CV_8UC1);
        temp_Mat.convertTo(temp, CV_8UC1);
        // cout<<"122 reached"<<endl;
        //result matrix to store I values
        Mat result = Mat::zeros(src.rows - temp.rows +1,src.cols - temp.cols +1, CV_32FC1);
        cout<<result.size<<endl;
        // cout<<"126 reached"<<endl;
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
        cout<<result.colRange(0,20).rowRange(0,20)<<endl;
        
        // cout <<"result: "<<result<<endl;

        //draw a rectangle on source image at the position where mutial information is the highest
        Point pt_max;
        double val_max, val_min;
        minMaxLoc(result, &val_max, &val_min, NULL, &pt_max);
        cout<<val_max<<endl;
        cout<<val_min<<endl;
        Mat dst;
        src.copyTo(dst);
        rectangle(dst, Rect(pt_max.x, pt_max.y, temp.cols, temp.rows), Scalar(255, 0, 255), 2, LINE_AA);

        //scale the value in result matrix to 0 to 255
        // normalize(result, result, 0, 255, NORM_MINMAX);

        //display the images
        imshow("src", src);
        imshow("temp", temp);
        imshow("match",dst);
        imshow("result",result);

        waitKey(0);
    }
}