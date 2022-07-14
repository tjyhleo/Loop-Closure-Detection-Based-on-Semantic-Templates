#include <iostream>
#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/imgcodecs.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <set>

using namespace std;
using namespace cv;

//function to calculate mutual information :I 
//  src: part of source image that overlapps with template image
//  templ: template image
//  templ_class: the set that contains the unique values in template image
float MI(const Mat src, const Mat templ, const set<float> templ_class)
    {
        float I=0.;
        
        set <float> src_class; //the set that contains the unique values in source image

        for (int i=0; i<src.rows; ++i)
        {
            for (int j=0; j<src.cols; ++j)
            {
                float value = src.at<float>(i,j);
                src_class.insert(value);
            }
        }

        //matrix that contains number of all combinations of values in template image and source image
        Mat his = Mat::zeros(src_class.size(),templ_class.size(),CV_32FC1);

        //go through source image and template image, add the value of each pixel to corresponding position in matrix his
        for (int i=0; i<src.rows; i++)
                {
                    for (int j=0; j<src.cols; j++)
                    {
                        float src_value = src.at<float>(i,j);
                        float templ_value = templ.at<float>(i,j);
                        int src_index, templ_index;
                        for (set<float>::iterator it = src_class.begin(); it != src_class.end(); it++)
                        {
                            if (*it==src_value)
                            {
                                for (set<float>::iterator iter = templ_class.begin(); iter != templ_class.end(); iter++)
                                {
                                    if (*iter==templ_value)
                                    {
                                        src_index = std::distance(src_class.begin(),it);
                                        templ_index = std::distance(templ_class.begin(),iter);
                                        his.at<float>(src_index,templ_index) += 1;
                                        break;

                                    }

                                }
                                break;
                            }
                        }
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

        return I;
    }

int main(int argc, char **argv)
{
    //read source image and template image
    Mat src_Mat = imread("/home/jialin/Pictures/semantic-src.png", IMREAD_GRAYSCALE);
    Mat templ_Mat = imread("/home/jialin/Pictures/semantic-template.png", IMREAD_GRAYSCALE);
    if (src_Mat.empty() || templ_Mat.empty())
    {
        cout << "failed to read image" << endl;
        return EXIT_FAILURE;
    }
  
    //convert data type to CV_32FC1
    Mat src(src_Mat.size(),CV_32FC1);
    src_Mat.convertTo(src, CV_32FC1);
    Mat templ(templ_Mat.size(),CV_32FC1);
    templ_Mat.convertTo(templ, CV_32FC1);

    //result matrix to store I values
    Mat result = Mat::zeros(src.cols - templ.cols +1, src.rows - src.rows +1, CV_32FC1);
    
    //the set that contains the unique values in template image
    set <float> templ_class;

    for (int i=0; i<templ.rows; ++i)
    {
        for (int j=0; j<templ.cols; ++j)
        {
            float value = templ.at<float>(i,j);
            templ_class.insert(value);
        }
    }

    //calculate I for each element of result
    for (int i=0; i<result.rows; i++)
    {
        for (int j=0; j<result.cols; j++)
        {
            Mat src_part(src, cv::Rect(i,j,templ.cols, templ.rows));
            float I = MI(src_part, templ, templ_class);
            result.at<float>(i,j)=I;
        }
    }
    
    // cout <<"result: "<<result<<endl;

    //draw a rectangle on source image at the position where mutial information is the highest
    Point pt_max;
    minMaxLoc(result, NULL, NULL, NULL, &pt_max);
    Mat dst;
    src.copyTo(dst);
    rectangle(dst, Rect(pt_max.x, pt_max.y, templ.cols, templ.rows), Scalar(255, 0, 255), 2, LINE_AA);

    //scale the value in result matrix to 0 to 255
    normalize(result, result, 0, 255, NORM_MINMAX);

    //display the images
    imshow("src", src);
    imshow("templ", templ);
    imshow("match",dst);
    imshow("result",result);

    waitKey(0);
}