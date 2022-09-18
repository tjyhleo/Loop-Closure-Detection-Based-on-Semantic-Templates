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
    int dynamic[]{24,25,26,27,28,29,30,31,32,33};
    // vector<int> dynamic_vec(dynamic, dynamic+10);
    ignoreLabels.insert(ignoreLabels.end(),dynamic,dynamic+10);
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

bool templateFinder(vector<vector<vector<uchar>>>& coarseTemp, 
                    Mat img, Mat ref_Mat, const int w, const int h, const vector<int> coarseLabels){
    bool templateOkay = false;
    int imgH = ref_Mat.rows;
    int imgW = ref_Mat.cols;
    vector<vector<uchar>> vvTempLast;
    for(int j=0; j<imgH - h; j+=imgH/40){
        vector<vector<uchar>> vvTemp;
        vector<uchar> vTempLast;
        for (int i=0; i<imgW - w; i+=imgW/200){
            Rect r(i,j, w,h);
            // if(8>7){
            //     Mat sth = img.clone();
            //     rectangle(sth, r, Scalar(255, 0, 255), 2, LINE_AA);
            //     imshow("template_rgb", sth);
            //     waitKey(0);
            // }
            

            //get the number of each type of semantic labels in template
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
            if(sem_set.size()<2){
                continue;
            }
            // //there must be no ignore semantic labels in template
            // bool ignoreLabelExist=false;
            // for(set<uchar>::iterator it = sem_set.begin(); it!=sem_set.end(); it++){
            //     if(find(ignoreLabels.begin(), ignoreLabels.end(), *it) != ignoreLabels.end()){
            //         ignoreLabelExist=true;
            //         break;
            //     };
            // }
            // if(ignoreLabelExist==true){
            //     continue;
            // }
            //there must be at least 3 semantic labels that each occupy more than 20% of the template
            int counter=0;
            set<uchar> major_sem; //record the semantic labels that occupy more than 20% of template
            for(set<uchar>::iterator it = sem_set.begin(); it!=sem_set.end(); it++){
                int num = count(sem_vec.begin(), sem_vec.end(), *it);
                if (num > w*h*0.3){
                    counter+=1;
                    major_sem.insert(*it);
                }
            }
            if (counter<2){
                // templateOkay=false;
                continue;
            }

            else if(counter==2){
                //if any the 2 major semantic labels are in coarseLabels, skip
                bool FoundCoarseLabel=false;
                for(set<uchar>::iterator it = major_sem.begin(); it!=major_sem.end(); it++){
                    if(find(coarseLabels.begin(), coarseLabels.end(), *it) != coarseLabels.end()){
                        FoundCoarseLabel=true;
                        break;
                    };
                }
                if(FoundCoarseLabel==true){
                    continue;
                }

                vector<uchar> vTemp;
                for(set<uchar>::iterator it=major_sem.begin(); it!=major_sem.end(); it++){
                    vTemp.push_back(*it);
                }
                if(vTemp.empty()){
                    continue;
                }
                if(vTempLast.empty()||vTemp!=vTempLast){
                    vvTemp.push_back(vTemp);
                    vTempLast=vTemp;
                    rectangle(img, r, Scalar(255, 255, 255), 2, LINE_AA);
                }
                else if (vTemp==vTempLast)
                {
                    continue;
                }
            }
        }

        if(vvTemp.empty()){
            continue;
        }
        if(vvTempLast.empty()||vvTemp!=vvTempLast){
            coarseTemp.push_back(vvTemp);
            vvTempLast=vvTemp;
            // for(size_t i=0; i<vvTempLast.size(); i++){
            //     for(size_t j=0; j<vvTempLast[i].size(); j++){
            //         cout<<int(vvTempLast[i][j])<<"  ";
            //     }
            //     cout<<",   ";
            // }
            // cout<<endl;
            // imshow("template_rgb", img);
            // waitKey(0);
        }
        else if(vvTemp==vvTempLast){
            continue;
        }
    }

    if(!coarseTemp.empty()){
        templateOkay=true;
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
    badLabels.push_back(22);
    return badLabels;
}

int resultFilter(Mat inputMat, set<int>essential_sem){
    int result=1;
    for(set<int>::iterator it=essential_sem.begin(); it!=essential_sem.end(); it++){
        bool foundIt=false;
        for(int r=0; r<inputMat.rows; r++){
            for(int c=0; c<inputMat.cols; c++){
                if(inputMat.at<uchar>(r,c)==*it){
                    foundIt=true;
                    // cout<<"found it: "<<int(*it)<<endl;
                    break;
                }
            }
            if(foundIt==true){
                break;
            }
        }
        if(foundIt==false){
            result=0;
            break;
        }
    }
    return result;
}

int TempScoreInner(vector<vector<uchar>> const vvRef, vector<vector<uchar>> const vvSrc){
    int score=0;
    if(vvRef.size()<2 || vvSrc.size()<2){
        return score;
    }
    size_t maxSize = max(vvRef.size(), vvSrc.size());
    size_t minSize = min(vvRef.size(), vvSrc.size());
    size_t size_diff = maxSize - minSize;
    vector<vector<uchar>> vvMax, vvMin;
    if(vvRef.size()==maxSize){
        vvMax=vvRef;
        vvMin=vvSrc;
    }
    else{
        vvMax=vvSrc;
        vvMin=vvRef;
    }
    bool matchFound=false;
    for(size_t s=0; s<vvMin.size()-1; s++){
        for(size_t ss=0; ss<s+1; ss++){
            vector<vector<uchar>> vvMinPart{vvMin.begin()+ss, vvMin.end()-(s-ss)};
            for(size_t sss=0; sss<s+1; sss++){
                vector<vector<uchar>> vvMaxPart{vvMax.begin()+sss, vvMax.end()-(s-sss)-size_diff};
                if(vvMaxPart.size()!=vvMinPart.size()){
                    cout<<"TempScoreInner got vector size wrong"<<endl;
                    exit(0);
                }
                if(vvMaxPart==vvMinPart){
                    score = int(vvMaxPart.size());
                    matchFound=true;
                    break;
                }
            }
            if(matchFound==true){
                break;
            }
        }
        if(matchFound==true){
            break;
        }
    }
    return score;
}

int MaxComb(Mat inMat){
    assert(inMat.type()==4 && "input mat type must be int");
    int score=0;
    if(inMat.rows==1 || inMat.cols==1){
        double val_max, val_min;
        minMaxLoc(inMat, &val_min, &val_max, NULL, NULL);
        score=int(val_max);
        return score;
    }

    int RUCorner = inMat.at<int>(0,inMat.cols-1);
    int LDCorner = inMat.at<int>(inMat.rows-1, 0);
    if(RUCorner>score){
        score=RUCorner;
    } 
    if(LDCorner>score){
        score=LDCorner;
    }
    for(int i=0; i<inMat.cols-1; i++){
        Mat innerMat = inMat.rowRange(1,inMat.rows).colRange(i+1,inMat.cols);
        int maxFromMat = MaxComb(innerMat);
        int col_score = inMat.at<int>(0,i) + maxFromMat;
        if(col_score>score){
            score=col_score;
        }
    }
    for(int i=0; i<inMat.rows-1; i++){
        Mat innerMat = inMat.rowRange(i+1,inMat.rows).colRange(1,inMat.cols);
        int maxFromMat = MaxComb(innerMat);
        int row_score = inMat.at<int>(i,0) + maxFromMat;
        if(row_score>score){
            score=row_score;
        }
    }
    return score;
}

int TempScore(vector<vector<vector<uchar>>> const vvvRef, vector<vector<vector<uchar>>> const vvvSrc){
    int score=0;
    // cout<<"building matrix..."<<endl;
    Mat mScore = Mat::zeros(vvvRef.size(), vvvSrc.size(), CV_32SC1);
    for(size_t i=0; i<vvvSrc.size(); i++){
        vector<vector<uchar>> vvSrc = vvvSrc[i];
        for(size_t j=0; j<vvvRef.size(); j++){
            vector<vector<uchar>> vvRef = vvvRef[j];
            int innerScore = TempScoreInner(vvRef, vvSrc);
            mScore.at<int>(j,i) = innerScore;
        }
    }
    // cout<<"getting max combination from matrix...  "<<mScore.rows<<", "<<mScore.cols<<endl;
    if(mScore.cols>14){
        mScore=mScore.colRange(0,14);
    }
    if(mScore.rows>14){
        mScore=mScore.rowRange(0,14);
    }
    score = MaxComb(mScore);

    return score;
}

int main(int argc, char **argv)
{
    clock_t start, end; //timing
    double t_diff; //timing
    float ConPro; //conditional probability
    Point pt_max, pt_min;
    double val_max, val_min;
    int imgSkip=0;
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
    vector<int> coarseLabels{6,7,8,9,10,11,12,21,22,23};

    //convert probability matrix to log of probability matrix
    log(ConProMat, ConProMat);



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
    for(int i=0; i<2680; i+=5){
        string kw = "0000001261";
        int str_len = fn_seg[i].length();
        string frameId = fn_seg[i].substr(str_len-14, 10);
        if(kw!=frameId){
            continue;
        }
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

        int w = 20;
        int h = 10;
        int x,y;
        start = clock();
        cout<<"extracting reference templates... "<<endl;
        vector<vector<vector<uchar>>> coarseTemp;
        bool templateOkay = templateFinder(coarseTemp,ref_segrgb, ref_Mat, w, h, ignoreLabels);
        end=clock();
        t_diff=(double)(end-start)/CLOCKS_PER_SEC;
        cout<<"time to find template: "<<t_diff<<endl;
        if(templateOkay==false){
            cout<<"image"<<to_string(i)<<"doesn't have a good template"<<endl;
            imgSkip+=1;
            cout<<"skipped img number: "<<imgSkip<<endl;
            continue;
        }
        
        // waitKey(0);

        
        for(int ii=0; ii<10; ii++){
            Mat src_Mat = imread(fn_seg[i-ii*1], IMREAD_GRAYSCALE);
            // Mat src_color = imread(fn_raw[i+1], IMREAD_COLOR);
            Mat src_segrgb = imread(fn_segrgb[i-ii*1],IMREAD_COLOR);
            if (src_Mat.empty() || src_segrgb.empty())
            {
                cout << "failed to read image" << endl;
                return EXIT_FAILURE;
            }

            cout<<"extracting src templates... "<<endl;
            vector<vector<vector<uchar>>> srcTemp;
            bool templateOkay = templateFinder(srcTemp,src_segrgb, src_Mat, w, h, ignoreLabels);
            if(templateOkay==false){
                cout<<"image"<<to_string(i)<<"doesn't have a good template"<<endl;
                imgSkip+=1;
                cout<<"skipped img number: "<<imgSkip<<endl;
                continue;
            }
            // waitKey(0);
            cout<<"calculating similarity... "<<endl;
            int score = TempScore(coarseTemp, srcTemp);
            cout<<"similarity:     "<<score<<endl;
        }
        break;
    }
    //     for(size_t i=0; i<vvTemplates.size(); i++){
    //         x=vvTemplates[i][0];
    //         y=vvTemplates[i][1];
    //         set<int> essential_sem; //stores the essential semantic labels in template, the matched template must have these semantic labels
    //         essential_sem.insert(vvTemplates[i][2]);
    //         essential_sem.insert(vvTemplates[i][3]);
    //         essential_sem.insert(vvTemplates[i][4]);
        
    //         Rect r(x,y, w,h);
    //         Mat temp_Mat = ref_Mat(r);
    //         Mat src(src_Mat.size(),CV_8UC1);
    //         src_Mat.convertTo(src, CV_8UC1);
    //         Mat temp(temp_Mat.size(),CV_8UC1);
    //         temp_Mat.convertTo(temp, CV_8UC1);
    //         Mat result = Mat::zeros(src.rows - temp.rows +1,src.cols - temp.cols +1, CV_32FC1);
    //         Mat result_filter = Mat::zeros(src.rows - temp.rows +1,src.cols - temp.cols +1, CV_32SC1);


    //         //show template in rgb image
    //         // rectangle(ref_color, r, Scalar(255, 0, 255), 2, LINE_AA);
    //         // imshow("template_color", ref_color);
    //         // rectangle(ref_segrgb, r, Scalar(0, 0, 255), 2, LINE_AA);
    //         // imshow("template_rgb", ref_segrgb);
    //         // waitKey(0);
    //         // cout<<"program continues"<<endl;


    //         // //show the number of each semantic label in template
    //         // showSemNum(temp);


    //         //calculate ConPro for each element of result matrix
            
    //         for (int i=0; i<result.rows; i++)
    //         {
    //             for (int j=0; j<result.cols; j++)
    //             {
    //                 Mat src_part(src, cv::Rect(j,i,temp.cols, temp.rows));
    //                 ConPro = ConPro_calculator(src_part, temp, ConProMat);
    //                 result.at<float>(i,j) = ConPro;
    //                 int fil=resultFilter(src_part, essential_sem);
    //                 // if(fil==1){
    //                 //     cout<<fil<<endl;
    //                 // }
    //                 result_filter.at<int>(i,j) = fil;
    //                 // cout<<src_part.rowRange(0,5).colRange(0,5)<<endl;
    //                 // exit(0);
                    
    //             }
    //         }
            

    //         //draw a rectangle on source image at the position where mutial information is the highest
    //         Mat mask = result_filter>0;
    //         cout<<"this is reached"<<endl;
    //         // cout<<mask.rowRange(100,105).colRange(565,570)<<endl;
            
    //         // Mat filtered_result;
    //         // result.copyTo(filtered_result,mask);
    //         // cout<<filtered_result.rowRange(100,105).colRange(565,570)<<endl;
    //         // cout<<filtered_result.rowRange(0,5).colRange(0,5)<<endl;
    //         minMaxLoc(result, &val_min, &val_max, NULL, &pt_max,mask);
    //         // cout<<result.rowRange(pt_max.y,pt_max.y+10).colRange(pt_max.x,pt_max.x+10)<<endl;
    //         // cout<<val_max<<endl;
    //         // cout<<val_min<<endl;
    //         cout<<pt_max.x<<" , "<<pt_max.y<<endl;
            
    //         // if(abs(x-pt_max.x)>50 || abs(y-pt_max.y)>50){
    //         if(8>7){
    //             mismatch_vec.push_back(i);
    //             // cout<<"match too far away: "<<frameId<<endl;
    //             // cout<<abs(x-pt_max.x)<<", "<<abs(y-pt_max.y)<<endl;
    //             // rectangle(ref_segrgb, r, Scalar(0, 0, 255), 2, LINE_AA);
    //             rectangle(src_segrgb, Rect(pt_max.x, pt_max.y, temp.cols, temp.rows), Scalar(0, 0, 255), 2, LINE_AA);
    //             float threshold = float((val_max-val_min)*0.99+val_min);


    //             // cout<<"ManualCheck for ground truth position"<<endl;
    //             // Mat srcPart = src(r);
    //             // ManualCheck(temp, srcPart, ConProMat);

    //             // cout<<"ManualCheck for best match position"<<endl;
    //             // srcPart = src(Rect(pt_max.x, pt_max.y, w,h));
    //             // ManualCheck(temp, srcPart, ConProMat);
    //             // waitKey(0); 


    //             for(int i=0; i<result.rows; i++){
    //                 for(int j=0; j<result.cols; j++){
    //                     if(result.at<float>(i,j)>threshold){
    //                         // drawMarker(src_color, Point(j,i), Scalar(0,0,255), MARKER_TILTED_CROSS, 5, 1,8);
    //                         drawMarker(src_segrgb, Point(j,i), Scalar(255,255,255), MARKER_TILTED_CROSS, 5, 1,8);
    //                     }
    //                     }
    //             }
                
   
    //             Mat heatMap;
    //             normalize(result, heatMap, 0, 255, NORM_MINMAX);
    //             heatMap.convertTo(heatMap, CV_8UC1);
    //             applyColorMap(heatMap, heatMap, COLORMAP_JET);


    //             // vector<int> compression_params;
    //             // compression_params.push_back(IMWRITE_PNG_COMPRESSION);
    //             // compression_params.push_back(3);
    //             // bool flag = false;
    //             // // flag = imwrite(kitti360 + "MI_images/"+frameId+"_color_temp.png", ref_color, compression_params);
    //             // // flag = imwrite(kitti360 + "MI_images/"+frameId+"_color_match.png", src_color, compression_params);
    //             // flag = imwrite(kitti360 + "MI_images/strange_images/"+frameId+"_seg_temp.png", ref_segrgb, compression_params);
    //             // flag = imwrite(kitti360 + "MI_images/strange_images/"+frameId+"_seg_match.png", src_segrgb, compression_params);
    //             // flag = imwrite(kitti360 + "MI_images/strange_images/"+frameId+"_heatMap.png", heatMap, compression_params);
    //             // imshow("template_rgb", ref_segrgb);
    //             imshow("heatmap", heatMap);
    //             imshow("src_rgb", src_segrgb);
    //             waitKey(0);
    //         }
    //     }
    //     vector<int> compression_params;
    //     compression_params.push_back(IMWRITE_PNG_COMPRESSION);
    //     compression_params.push_back(3);
    //     bool flag = false;
    //     // flag = imwrite(kitti360 + "MI_images/"+frameId+"_color_temp.png", ref_color, compression_params);
    //     // flag = imwrite(kitti360 + "MI_images/"+frameId+"_color_match.png", src_color, compression_params);
    //     flag = imwrite(kitti360 + "MI_images/strange_images/"+frameId+"_seg_temp.png", ref_segrgb, compression_params);
    //     flag = imwrite(kitti360 + "MI_images/strange_images/"+frameId+"_seg_match.png", src_segrgb, compression_params);
    //     // flag = imwrite(kitti360 + "MI_images/strange_images/"+frameId+"_heatMap.png", heatMap, compression_params);
    // }
    // // for(int i=0; i<mismatch_vec.size(); i++){
    // //         cout<<i<<endl;
    // // }
}