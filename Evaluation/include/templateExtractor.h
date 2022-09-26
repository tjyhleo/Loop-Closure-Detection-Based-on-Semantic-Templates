#ifndef TEMPLATEEXTRACTOR_H
#define TEMPLATEEXTRACTOR_H

#include <iostream>
#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/imgcodecs.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <set>
#include <ctime>
#include <numeric>
using namespace std;
using namespace cv;

void get_histMat(Mat &histMat);
vector<int> get_ignoreLabels();
bool checkTempPattern(Mat temp_Mat);
bool nms(vector<int> currentP, vector<vector<int>> Temp, int w, int h);
bool templateExtractor(vector<vector<int>>& coarseTemp,vector<vector<int>>& fineTemp, 
                Mat ref_Mat, const int w, const int h, const vector<int> ignoreLabels);

#endif