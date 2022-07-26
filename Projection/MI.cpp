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
void MI(const Mat src, const Mat temp, const Mat histMat, Mat& dst)
    {
        float I=0.;

        //go through source image and template image, add the value of each pixel to corresponding position in matrix his
        for (int i=0; i<temp.rows; i++)
            {
                for (int j=0; j<temp.cols; j++)
                {
                    
                }
            }
        
        float total = sum(his)[0]; //sum up all the elements in his

        //for each element in his, calculate: Pst*log(Pst/(Ps*Pt)), and add its value to I
        for (int i=0; i<his.rows; i++)
        {
            for (int j=0; j<his.cols; j++)
            {
                float Ps=0., Pt=0., Pst=0.;
                Pst = his.at<float>(i,j);
                if (Pst==0)
                {
                    Pst=1;
                }
                for(int ii=0; ii<his.rows; ii++)
                {
                    Pt+=his.at<float>(ii,j);
                }
                for(int jj=0; jj<his.rows; jj++)
                {
                    Ps+=his.at<float>(i,jj);
                }

                I+=Pst/total * log(Pst*total/(Ps*Pt));
                
            }
        }

    }

int main(int argc, char **argv)
{
    clock_t start, end; //timing
    double t_diff; //timing
    int tempWl = 200;   //template width
    int tempWr = 1200;
    int tempHu = 50;   //template hight
    int tempHd = 300;

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


    //get all the picture names in segmentation folder
    string seg_path=kitti360+"2013_05_28_drive_0007_sync_image_00/segmentation";
    vector<cv::String> fn_seg;
    glob(seg_path, fn_seg, false);
    for(int i=0; i<fn_seg.size()-1; i++){
        
        Mat ref_Mat = imread(fn_seg[i], IMREAD_GRAYSCALE);
        Mat src_Mat = imread(fn_seg[i+1], IMREAD_GRAYSCALE);
        Rect r(tempWl, tempHu, tempWr, tempHd);
        Mat temp_Mat = ref_Mat(r);
        if (src_Mat.empty() || temp_Mat.empty())
        {
            cout << "failed to read image" << endl;
            return EXIT_FAILURE;
        }
    
        //convert data type to CV_32FC1
        Mat src(src_Mat.size(),CV_32FC1);
        src_Mat.convertTo(src, CV_32FC1);
        Mat temp(temp_Mat.size(),CV_32FC1);
        temp_Mat.convertTo(temp, CV_32FC1);

        //result matrix to store I values
        Mat result = Mat::zeros(src.cols - temp.cols +1, src.rows - src.rows +1, CV_32FC1);

        //calculate I for each element of result
        for (int i=0; i<result.rows; i++)
        {
            for (int j=0; j<result.cols; j++)
            {
                Mat src_part(src, cv::Rect(i,j,temp.cols, temp.rows));
                MI(src_part, temp, histMat, result);
            }
        }
        
        // cout <<"result: "<<result<<endl;

        //draw a rectangle on source image at the position where mutial information is the highest
        Point pt_max;
        minMaxLoc(result, NULL, NULL, NULL, &pt_max);
        Mat dst;
        src.copyTo(dst);
        rectangle(dst, Rect(pt_max.x, pt_max.y, temp.cols, temp.rows), Scalar(255, 0, 255), 2, LINE_AA);

        //scale the value in result matrix to 0 to 255
        normalize(result, result, 0, 255, NORM_MINMAX);

        //display the images
        imshow("src", src);
        imshow("temp", temp);
        imshow("match",dst);
        imshow("result",result);

        waitKey(0);
    }
}