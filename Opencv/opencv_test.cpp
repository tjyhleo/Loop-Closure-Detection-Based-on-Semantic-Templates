#include<iostream>
#include<opencv4/opencv2/opencv.hpp>
#include<opencv4/opencv2/core/core.hpp>
// #include<opencv4/opencv2/highgui.hpp>
#include<opencv4/opencv2/opencv.hpp>
 
 
// #define Usage()\
// {std::cerr<<"usage: ./showpic FILE"<<std::endl;}
// int main (int argc, char **argv)
// {
//   if(argc !=2) Usage();
//   cv::Mat img=cv::imread("/home/jialin/VSC_Projects/Opencv/1.png");
//   cv::imshow("window",img);
//   cv::waitKey(0);
//   return 0;
// }

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
  Mat src = imread("/home/jialin/VSC_Projects/Opencv/2.png");
  imshow("input", src);
  waitKey(0);
  destroyAllWindows();
  return 0;

}