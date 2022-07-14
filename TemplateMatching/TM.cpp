#include <iostream>
#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/imgcodecs.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/highgui.hpp>

using namespace std;
using namespace cv;

int main(int argc, char **argv)
{
    Mat src = imread("/home/jialin/Pictures/source.png", IMREAD_COLOR);
    Mat templ = imread("/home/jialin/Pictures/template.png", IMREAD_COLOR);
    if (src.empty() || templ.empty())
    {
        cout << "failed to read image" << endl;
        return EXIT_FAILURE;
    }
    cout << "src.cols:" << src.cols << endl;
    cout << "src.rows:" << src.rows << endl;
    cout << "templ.cols:" << templ.cols << endl;
    cout << "templ.rows:" << templ.rows << endl;


    vector<int> methods = {TM_SQDIFF, TM_SQDIFF_NORMED, TM_CCORR, TM_CCORR_NORMED, TM_CCOEFF, TM_CCOEFF_NORMED};
    vector<String> method_names = {"sqdiff", "sqdiff_norm", "ccorr", "ccorr_norm", "ccoeff", "ccoeff_normed"};
    vector<Mat> match_mats(methods.size());
    for (int i =0; i < methods.size(); i++)
    {
        Mat &match_mat = match_mats[i];
        matchTemplate(src, templ, match_mat, methods[i]);


    }

    vector<Point> match_locations(methods.size());
    for (int i = 0; i < methods.size(); i++){
        Point pt_min, pt_max;
        double val_min, val_max;
        minMaxLoc(match_mats[i], &val_min, &val_max, &pt_min, &pt_max);
        if (methods[i] == TM_SQDIFF || methods[i] == TM_SQDIFF_NORMED) {
            match_locations[i] = pt_min;
        }
        else {
            match_locations[i] = pt_max;
        }
        cout << method_names[i] << " = " << match_locations[i] << endl;
        cout << method_names[i] << " cols " << match_mats[i].cols << " rows " << match_mats[i].rows << endl;
        // cout << method_names[i] << "  max  " << val_max <<endl;
        // cout << method_names[i] << "  min  " << val_min << endl;


        }
    
    vector<Mat> match_normed_mats(methods.size());
    for (int i = 0; i < methods.size(); i++){
        Mat& match_normed_mat = match_normed_mats[i];
        normalize(match_mats[i], match_normed_mat, 0, 255, NORM_MINMAX);
    }

    vector<Mat> dsts(methods.size());
    for (int i = 0; i < methods.size(); i++){
        Mat& dst = dsts[i];
        src.copyTo(dst);
        Point match_loc = match_locations[i];
        rectangle(dst, Rect(match_loc.x, match_loc.y, templ.cols, templ.rows), Scalar(255, 0, 255), 2, LINE_AA);

    }

    imshow("src", src);
    imshow("templ", templ);
    for (int i = 0; i < methods.size(); i++){
        imshow(method_names[i], match_mats[i]);
        imshow(method_names[i] + "_dst", dsts[i]);

    }
    waitKey(0);

}