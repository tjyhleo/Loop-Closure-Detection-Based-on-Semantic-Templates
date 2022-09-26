#ifndef IMGCOMPARE_H
#define IMGCOMPARE_H

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


bool findGroup(vector<vector<int>>& tempGroup, vector<vector<int>> allTemplates, vector<int>targetLabels);
bool matchGroup(vector<vector<int>>qryGroup, vector<vector<int>>refGroup);
int coarseMatch(vector<vector<int>> qryTemplates, vector<vector<int>> refTemplates, Mat qryImg_rgb, Mat refImg_rgb, int w, int h, bool display);
int fineMatch(vector<vector<int>> qryTemplates, vector<vector<int>> refTemplates);
float imgCompare(vector<vector<int>> fineTemplates1,
                vector<vector<int>> coarseTemplates1,
                vector<vector<int>> fineTemplates2,
                vector<vector<int>> coarseTemplates2);


#endif