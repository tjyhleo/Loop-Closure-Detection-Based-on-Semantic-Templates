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
    // cout<<"ignore labels:   ";
    for(int i = 0; i<ConProMat.rows; i++){
        float s = ConProMat.at<float>(i,i);
        if(s<0.6){
            ignoreLabels.push_back(i);
            // cout<<i<<": "<<s<<"     ";
        }
    }
    // cout<<endl;
    // int dynamic[]{24,25,26,27,28,29,30,31,32,33};
    // ignoreLabels.insert(ignoreLabels.end(),dynamic,dynamic+10);
    // int terrain =22;
    int dynamic[]{11,12,13,14,15,16,17,18};
    ignoreLabels.insert(ignoreLabels.end(),dynamic,dynamic+8);
    int terrain =9;
    ignoreLabels.push_back(terrain);
    return ignoreLabels;
}


void get_histMat(Mat &histMat){
    FileStorage fs_read;
    double val_min, val_max;
    fs_read.open("histMat_unS.xml", FileStorage::READ);
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


bool checkTempPattern(Mat temp_Mat){
    set<uchar> left, right, up, down;
    set<uchar> left_set, right_set, up_set, down_set;
    vector<uchar> left_vec, right_vec, up_vec, down_vec;
    bool bad_pattern = false;
    
    for(int i=0; i<temp_Mat.rows; i++){
        left_set.insert(temp_Mat.at<uchar>(i,0));
        left_vec.push_back(temp_Mat.at<uchar>(i,0));
        right_set.insert(temp_Mat.at<uchar>(i,temp_Mat.cols-1));
        right_vec.push_back(temp_Mat.at<uchar>(i,temp_Mat.cols-1));
    }

    for(int i=0; i<temp_Mat.cols; i++){
        up_set.insert(temp_Mat.at<uchar>(0,i));
        up_vec.push_back(temp_Mat.at<uchar>(0,i));
        down_set.insert(temp_Mat.at<uchar>(temp_Mat.rows-1, i));
        down_vec.push_back(temp_Mat.at<uchar>(temp_Mat.rows-1, i));
    }

    for(set<uchar>::iterator it=left_set.begin(); it!=left_set.end(); it++){
        if(count(left_vec.begin(),left_vec.end(), *it)>left_vec.size()/left_set.size()*0.3){
            left.insert(*it);
        }
    }

    for(set<uchar>::iterator it=right_set.begin(); it!=right_set.end(); it++){
        if(count(right_vec.begin(),right_vec.end(), *it)>right_vec.size()/right_set.size()*0.3){
            right.insert(*it);
        }
    }

    for(set<uchar>::iterator it=up_set.begin(); it!=up_set.end(); it++){
        if(count(up_vec.begin(),up_vec.end(), *it)>up_vec.size()/up_set.size()*0.3){
            up.insert(*it);
        }
    }

    for(set<uchar>::iterator it=down_set.begin(); it!=down_set.end(); it++){
        if(count(down_vec.begin(),down_vec.end(), *it)>down_vec.size()/down_set.size()*0.3){
            down.insert(*it);
        }
    }

    if(up.size()==1 && down.size()==1){
        int counter=0;
        for(set<uchar>::iterator it=left.begin(); it!=left.end(); it++){
            if(find(right.begin(), right.end(), *it)!= right.end()){
                // bad_pattern=true;
                // break;
                counter+=1;
            }
        }
        if(counter>1){
            bad_pattern=true;
        }
    }

    if(bad_pattern==true){
        return bad_pattern;
    }

    if(left.size()==1 && right.size()==1){
        int counter=0;
        for(set<uchar>::iterator it=up.begin(); it!=up.end(); it++){
            if(find(down.begin(), down.end(), *it)!= down.end()){
                counter+=1;
                // bad_pattern=true;
                // break;
            }
        }
        if(counter>1){
            bad_pattern=true;
        }
    }

    if(bad_pattern==true){
        return bad_pattern;
    }

    cout<<left.size()<<","<<right.size()<<","<<up.size()<<","<<down.size()<<endl;
    
    for(set<uchar>::iterator it=left.begin(); it!=left.end(); it++){
        cout<<"left: "<<int(*it)<<": "<<count(left_vec.begin(),left_vec.end(), *it)<<",  ";
    }
    cout<<endl;
    for(set<uchar>::iterator it=right.begin(); it!=right.end(); it++){
        cout<<"right: "<<int(*it)<<": "<<count(right_vec.begin(),right_vec.end(), *it)<<",  ";
    }
    cout<<endl;
    for(set<uchar>::iterator it=up.begin(); it!=up.end(); it++){
        cout<<"up: "<<int(*it)<<": "<<count(up_vec.begin(),up_vec.end(), *it)<<",  ";
    }
    cout<<endl;
    for(set<uchar>::iterator it=down.begin(); it!=down.end(); it++){
        cout<<"down: "<<int(*it)<<": "<<count(down_vec.begin(),down_vec.end(), *it)<<",  ";
    }
    cout<<endl;

    return bad_pattern;
}


bool nms(vector<int> currentP, vector<vector<int>> Temp, int w, int h){
    //get the past point that's closest to the current point
    bool overlap=false;
    float min_diff;
    int i ,j;
    if(Temp.empty()){
        return overlap;
    }
    for(size_t t=0; t<Temp.size(); t++){
        vector<int>pastP{Temp[t][0], Temp[t][1]};
        float diff = norm(currentP, pastP);
        if(t==0 || diff<min_diff){
            min_diff=diff;
            i=Temp[t][0];
            j=Temp[t][1];
        }
    }
    //test if the past template may overlap with the current template
    if(abs(currentP[0]-i)<w && abs(currentP[1]-j)<h){
        overlap=true;
    }

    return overlap;
}


bool checkRepeatPattern(set<uchar> major_sem, vector<vector<int>> fineTemp){
    bool repeatPattern = false;
    set<int>current_sem;
    for(set<uchar>::iterator it=major_sem.begin(); it!=major_sem.end(); it++ ){
        current_sem.insert(int(*it));
    }
    
    for(size_t t=0; t<fineTemp.size(); t++){
        set<int>past_sem(fineTemp[t].begin()+2, fineTemp[t].end());
        if(current_sem==past_sem){
            repeatPattern=true;
            break;
        }
    }
    return repeatPattern;
}

bool templateFinder(vector<vector<int>>& coarseTemp,vector<vector<int>>& fineTemp, 
                    Mat img, Mat ref_Mat, const int w, const int h, const vector<int> ignoreLabels){
    bool templateOkay = false;
    bool lowerRequire=false;
    int imgH = ref_Mat.rows;
    int imgW = ref_Mat.cols;
    // int coarseXLast=-w, coarseYLast=-h, okayXLast=-w, okayYLast=-h;
    
    for(int j=0; j<imgH - h; j+=3){
        for (int i=0; i<imgW - w; i+=3){
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
            //there must be no ignore semantic labels in template
            bool ignoreLabelExist=false;
            for(set<uchar>::iterator it = sem_set.begin(); it!=sem_set.end(); it++){
                if(find(ignoreLabels.begin(), ignoreLabels.end(), *it) != ignoreLabels.end()){
                    ignoreLabelExist=true;
                    break;
                };
            }
            if(ignoreLabelExist==true){
                continue;
            }
            //there must be at least 3 semantic labels that each occupy more than 20% of the template
            int counter=0;
            set<uchar> major_sem; //record the semantic labels that occupy more than 20% of template
            for(set<uchar>::iterator it = sem_set.begin(); it!=sem_set.end(); it++){
                int num = count(sem_vec.begin(), sem_vec.end(), *it);
                if (num > w*h*0.15){
                    counter+=1;
                    major_sem.insert(*it);
                }
            }
            if (counter<2){
                // templateOkay=false;
                continue;
            }

            else if(counter==2){
                //确保每个label占比30%以上
                int counter=0;
                for(set<uchar>::iterator it = major_sem.begin(); it!=major_sem.end(); it++){
                    int num = count(sem_vec.begin(), sem_vec.end(), *it);
                    if (num > w*h*0.4){
                        counter+=1;
                    }
                }
                if(counter==2){
                    vector<int> xy {i,j};
                    bool overlap = nms(xy, coarseTemp, w, h);
                    if(overlap==true){
                        continue;
                    }
                    for(set<uchar>::iterator it=major_sem.begin(); it!=major_sem.end(); it++){
                        xy.push_back(int(*it));
                    }
                    coarseTemp.push_back(xy);
                    rectangle(img, r, Scalar(255, 255, 255), 1, LINE_AA);
                    // imshow("template_rgb", img);
                    // waitKey(0);
                }
            }

            else if(counter>2){
                vector<int>xy{i,j};
                bool overlap = nms(xy, fineTemp, w, h);
                if(overlap==true){
                    continue;
                }
                // bool badPattern = checkTempPattern(temp_Mat);
                // if(badPattern==true){
                //     continue;
                // }
                bool repeatPattern = checkRepeatPattern(major_sem, fineTemp);
                if(repeatPattern==true){
                    continue;
                }
                for(set<uchar>::iterator it=major_sem.begin(); it!=major_sem.end(); it++){
                    xy.push_back(int(*it));
                }
                fineTemp.push_back(xy);
                rectangle(img, r, Scalar(0, 0, 255), 1, LINE_AA);
                // imshow("template_rgb", img);
                // waitKey(0);
            }
        }
    }

    if(!coarseTemp.empty() || !fineTemp.empty()){
        templateOkay=true;
    }
    return templateOkay;
}

set<int> findMajorLabel(Mat temp, float threshold){
    vector<uchar> sem_vec;
    set<uchar> sem_set;
    for(int i=0; i<temp.rows; i++){
        for(int j=0; j<temp.cols; j++){
            sem_vec.push_back(temp.at<uchar>(i,j));
            sem_set.insert(temp.at<uchar>(i,j));
        }
    }
    int counter=0;
    set<int> major_sem; //record the semantic labels that occupy more than 20% of template
    for(set<uchar>::iterator it = sem_set.begin(); it!=sem_set.end(); it++){
        int num = count(sem_vec.begin(), sem_vec.end(), *it);
        if (num > temp.cols*temp.rows*threshold){
            counter+=1;
            major_sem.insert(int(*it));
        }
    }
    // for(set<uchar>::iterator it = template_labels_set.begin(); it!=template_labels_set.end(); it++){
    //     cout<<int(*it)<<" : "<<count(template_labels_vec.begin(), template_labels_vec.end(), *it)<<endl;
    // }
    return major_sem;

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
    // set<uchar> exist_sem;
    // for(int r=0; r<inputMat.rows; r++){
    //     for(int c=0; c<inputMat.cols; c++){
    //         exist_sem.insert(inputMat.at<uchar>(r,c));
    //     }
    // }
    // for(set<uchar>::iterator it=essential_sem.begin(); it!=essential_sem.end(); it++){
    //     if(find(exist_sem.begin(), exist_sem.end(), *it)==exist_sem.end()){
    //         result=0;
    //         break;
    //     }
    // }
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

template <typename T>
vector<size_t> sort_indexes_e(vector<T> &v)
{
    vector<size_t> idx(v.size());
    iota(idx.begin(), idx.end(), 0);
    sort(idx.begin(), idx.end(),
        [&v](size_t i1, size_t i2) {return v[i1] < v[i2]; });
    return idx;
}


void sortMatchedPoint(vector<int>& xVec, vector<int>& yVec, vector<int>& xRefVec, vector<int> &yRefVec,
                        const vector<vector<int>> fineMatechedTemp)
{
    xVec.clear();
    yVec.clear();
    xRefVec.clear();
    yRefVec.clear();
    if(!fineMatechedTemp.empty()){
        for(size_t t=0; t<fineMatechedTemp.size(); t++){
            xVec.push_back(fineMatechedTemp[t][0]);
            yVec.push_back(fineMatechedTemp[t][1]);
            xRefVec.push_back(fineMatechedTemp[t][2]);
            yRefVec.push_back(fineMatechedTemp[t][3]);
        }
        // vector<size_t> xSortIdx = sort_indexes_e(xVec);
        // vector<size_t> ySortIdx = sort_indexes_e(yVec);
        // vector<size_t> xRefSortIdx = sort_indexes_e(xRefVec);
        // vector<size_t> yRefSortIdx = sort_indexes_e(yRefVec);
        // if(xSortIdx!=xRefSortIdx || ySortIdx!=yRefSortIdx){
        //     cout<<"sort idx not identical"<<endl;
        //     for(int i=0; i<xVec.size(); i++){
        //             cout<<xVec[i]<<"  ";
        //     }
        //     cout<<endl;
        //     for(int i=0; i<xVec.size(); i++){
        //         cout<<yVec[i]<<"  ";
        //     }
        //     cout<<endl;
        //     for(int i=0; i<xVec.size(); i++){
        //         cout<<xRefVec[i]<<"  ";
        //     }
        //     cout<<endl;
        //     for(int i=0; i<xVec.size(); i++){
        //         cout<<yRefVec[i]<<"  ";
        //     }
        //     cout<<endl;
        //     // exit(0);
        // }
        sort(xVec.begin(), xVec.end());
        sort(yVec.begin(), yVec.end());
        sort(xRefVec.begin(), xRefVec.end());
        sort(yRefVec.begin(), yRefVec.end());
    }
}

bool checkFineMatch(vector<int> currentMatch, vector<vector<int>>pastMatch){
    bool fine =false;
    if(pastMatch.empty()){
        return fine;
    }
    if(pastMatch.size()<2){
        return fine;
    }
    size_t st = pastMatch.size();
    vector<int>currentP{currentMatch[0], currentMatch[1]};
    vector<int>currentPRef{currentMatch[2], currentMatch[3]};
    vector<int>lastP{pastMatch[st-1][0],pastMatch[st-1][1]};
    vector<int>lastPRef{pastMatch[st-1][2],pastMatch[st-1][3]};
    vector<int>last2P{pastMatch[st-2][0],pastMatch[st-2][1]};
    vector<int>last2PRef{pastMatch[st-2][2],pastMatch[st-2][3]};
    float refD = norm(currentPRef, lastPRef);
    float cD = norm(currentP, lastP);
    float refD2 = norm(currentPRef, last2PRef);
    float cD2 = norm(currentP, last2P);
    float refD3 = norm(last2PRef, lastPRef);
    float cD3 = norm(last2P, lastP);
    float ratio1=abs(cD/cD2 - refD/refD2)/(refD/refD2);
    float ratio2=abs(cD3/cD2 - refD3/refD2)/(refD3/refD2);
    float ratio3=abs(cD/cD3 - refD/refD3)/(refD/refD3);
    if(ratio1<0.2 && ratio2<0.2 && ratio3<0.2){
        fine=true;
    }
    return fine;
}

int main(int argc, char **argv)
{
    bool display=true;
    bool matchFineTemp=false;
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
    // vector<int> goodLabels = goodLabelFinder();
    // vector<int> badLabels = badLabelFinder();

    //convert probability matrix to log of probability matrix
    log(ConProMat, ConProMat);

    //get all the picture names in segmentation folder and raw image folder
    // string seg_path=kitti360+"data_2d_semantics/train/2013_05_28_drive_0010_sync/image_00/semantic";
    string seg_path = "/media/jialin/045E58135E57FC3C/UBUNTU/Evaluation_segmented/overcast-summer/rear/segmentation";
    // string seg_path=kitti360+"data_2d_raw/2013_05_28_drive_0010_sync/image_00/segmentation";
    vector<cv::String> fn_seg;
    glob(seg_path, fn_seg, false);


    string seg_path2 = "/media/jialin/045E58135E57FC3C/UBUNTU/Evaluation_segmented/night-rain/rear/segmentation";
    // string seg_path=kitti360+"data_2d_raw/2013_05_28_drive_0010_sync/image_00/segmentation";
    vector<cv::String> fn_seg2;
    glob(seg_path2, fn_seg2, false);

    // string raw_path = kitti360+"data_2d_raw/2013_05_28_drive_0010_sync/image_00/data_rect";
    string raw_path = "/media/jialin/045E58135E57FC3C/UBUNTU/Evaluation_segmented/overcast-summer/rear";
    vector<cv::String> fn_raw;
    glob(raw_path, fn_raw, false);

    // string segrgb_path = kitti360+"data_2d_raw/2013_05_28_drive_0010_sync/image_00/segmentation_rgb";
    // string segrgb_path=kitti360+"data_2d_semantics/train/2013_05_28_drive_0010_sync/image_00/semantic_rgb";
    string segrgb_path="/media/jialin/045E58135E57FC3C/UBUNTU/Evaluation_segmented/overcast-summer/rear/seg-rgb";
    vector<cv::String> fn_segrgb;
    glob(segrgb_path, fn_segrgb, false);


    string segrgb_path2 = "/media/jialin/045E58135E57FC3C/UBUNTU/Evaluation_segmented/night-rain/rear/seg-rgb";
    // string seg_path=kitti360+"data_2d_raw/2013_05_28_drive_0010_sync/image_00/segmentation";
    vector<cv::String> fn_segrgb2;
    glob(segrgb_path2, fn_segrgb2, false);


    //perform template matching image by image
    for(int i=0; i<fn_seg.size()-1; i++){
    // for(int i=0; i<2680; i+=5){
        // string kw = "0000002161";
        // int str_len = fn_seg[i].length();
        // string frameId = fn_seg[i].substr(str_len-14, 10);
        // if(kw!=frameId){
        //     continue;
        // }
        // cout<<"current image: "<<frameId<<endl;

        //read segmentation image and rgb image, select template and build result matrix
        Mat ref_Mat = imread(fn_seg[i], IMREAD_GRAYSCALE);
        // Mat src_Mat = imread(fn_seg[i+1], IMREAD_GRAYSCALE);
        Mat ref_color = imread(fn_raw[i], IMREAD_COLOR);
        // Mat src_color = imread(fn_raw[i+1], IMREAD_COLOR);
        Mat ref_segrgb = imread(fn_segrgb[i],IMREAD_COLOR);
        // Mat src_segrgb = imread(fn_segrgb[i+1],IMREAD_COLOR);
        if (ref_Mat.empty() || ref_color.empty() || ref_segrgb.empty())
        {
            cout << "failed to read image" << endl;
            return EXIT_FAILURE;
        }


        int w = 15;
        int h = 15;
        // int x,y;
        start = clock();
        
        vector<vector<int>> coarseTemplates;
        vector<vector<int>> fineTemplates;
        bool templateOkay = templateFinder(coarseTemplates, fineTemplates,ref_segrgb, ref_Mat, w, h, ignoreLabels);
        end=clock();
        t_diff=(double)(end-start)/CLOCKS_PER_SEC;
        cout<<"time to find template: "<<t_diff<<endl;
        if(templateOkay==false){
            cout<<"image"<<to_string(i)<<"doesn't have a good template"<<endl;
            imgSkip+=1;
            cout<<"skipped img number: "<<imgSkip<<endl;
            continue;
        }
        cout<<"fine template found: "<<fineTemplates.size()<<",    coarse template found: "<<coarseTemplates.size()<<endl;
        // exit(0);


        for(int ii=1; ii<8; ii++){
            start=clock();
            // Mat src_Mat = imread(fn_seg[i+17+ii*1], IMREAD_GRAYSCALE);
            // Mat src_segrgb = imread(fn_segrgb[i+17+ii*1],IMREAD_COLOR);
            Mat src_Mat = imread(fn_seg[ii], IMREAD_GRAYSCALE);
            Mat src_segrgb = imread(fn_segrgb[ii], IMREAD_COLOR);

            if (src_Mat.empty() || src_segrgb.empty())
            {
                cout << "failed to read image" << endl;
                return EXIT_FAILURE;
            }


            int fineMatchedNum=0;
            vector<vector<int>> fineMatchedPosition;

            if(matchFineTemp==true){
                for(size_t i=0; i<fineTemplates.size(); i++){
                    int xx=fineTemplates[i][0];
                    int yy=fineTemplates[i][1];
                    set<int> essential_sem; //stores the essential semantic labels in template, the matched template must have these semantic labels
                    for(size_t t = 2; t<fineTemplates[i].size(); t++){
                        essential_sem.insert(fineTemplates[i][t]);
                    }
                    
                    Rect r(xx,yy, w,h);
                    Mat temp_Mat = ref_Mat(r);
                    Mat src(src_Mat.size(),CV_8UC1);
                    src_Mat.convertTo(src, CV_8UC1);
                    Mat temp(temp_Mat.size(),CV_8UC1);
                    temp_Mat.convertTo(temp, CV_8UC1);
                    Mat result = Mat::zeros(src.rows - temp.rows +1,src.cols - temp.cols +1, CV_32FC1);
                    // Mat result_filter = Mat::zeros(src.rows - temp.rows +1,src.cols - temp.cols +1, CV_32SC1);

                    int yStart=0, xStart=0, yEnd=result.rows, xEnd=result.cols;
                    if(!fineMatchedPosition.empty()){
                        vector<int> lastPosition = fineMatchedPosition[fineMatchedPosition.size()-1];
                        int xMatched = lastPosition[0];
                        int yMatched = lastPosition[1];
                        // int iMatched = lastPosition[2];
                        int xLast = lastPosition[2];
                        int yLast = lastPosition[3];
                        //如果当前template在上一个matched template右边，就从上一个template matched的位置往右下边找
                        if(xx>xLast){
                            xStart=xMatched;
                            yStart=yMatched;
                            xEnd=result.cols;
                            yEnd=result.rows;
                        }
                        //如果当前template在上一个matched template左边，就从上一个template matched的位置往下找，从0找到template matched的位置
                        if(xx<xLast){
                            xStart=0;
                            yStart=yMatched;
                            xEnd=xMatched;
                            yEnd=result.rows;
                        }
                    }

                    bool tempMatched =false;
                    for (int y=yStart; y<yEnd; y++)
                    {
                        for (int x=xStart; x<xEnd; x++)
                        {
                            // Mat srcClone = src_segrgb.clone();
                            // Rect r(x,y, w,h);
                            // rectangle(srcClone, r, Scalar(255, 255, 255), 2, LINE_AA);
                            // imshow("refseg", ref_segrgb);
                            // imshow("template_rgb", srcClone);
                            // waitKey(0);
                            Mat src_part(src, cv::Rect(x,y,w,h));
                            set<int>major_sem = findMajorLabel(src_part, 0.15);
                            if(major_sem!=essential_sem){
                                continue;
                            }
                            vector<int> currentP{x,y};
                            bool overlap = nms(currentP, fineMatchedPosition, w, h);
                            if(overlap==true){
                                continue;
                            }
                            ConPro = ConPro_calculator(src_part, temp, ConProMat);
                            if(ConPro<log(0.3)*w*h){
                                continue;
                            }
                            // 先找到3个label的模板匹配，以这些label划出2个label的模板的搜索范围。
                            //然后以匹配到的2个label的模板，来划出下一个2个label的模板的搜索范围
                            
                            fineMatchedNum+=1;
                            vector<int> position{x,y,xx,yy};
                            fineMatchedPosition.push_back(position);

                            if(display==true){
                                Rect r(x,y, w,h);
                                Rect rr(xx,yy, w,h);
                                rectangle(ref_segrgb, rr, Scalar(0, 0, 255), 2, LINE_AA);
                                rectangle(src_segrgb, r, Scalar(0, 0, 255), 2, LINE_AA);
                                imshow("ref", ref_segrgb);
                                imshow("template", src_segrgb);
                                waitKey(0);
                            }
                            tempMatched=true;
                            break;
                        }
                        if(tempMatched==true){
                            break;
                        }
                    }
                }
                end=clock();
                t_diff=(double)(end-start)/CLOCKS_PER_SEC;
                cout<<"time taken for fineTemplate matching: "<<t_diff<<endl;
            }
            

            
            vector<int> xVec;
            vector<int> yVec;
            vector<int> xRefVec;
            vector<int> yRefVec;
            // sortMatchedPoint(xVec, yVec, xRefVec, yRefVec, fineMatchedPosition);
            int coarseMatchedNum=0;
            vector<vector<int>> coarseMatchedPosition;
            set<int> lastMajor_sem{0,0};
            vector<set<int>> deadSem;
            for(size_t i=0; i<coarseTemplates.size(); i++){
                int xx=coarseTemplates[i][0];
                int yy=coarseTemplates[i][1];
                set<int> essential_sem; //stores the essential semantic labels in template, the matched template must have these semantic labels
                for(size_t t = 2; t<coarseTemplates[i].size(); t++){
                    essential_sem.insert(coarseTemplates[i][t]);
                }

                //如果有7个以上该label找不到，那就不要找这个label了
                if(!deadSem.empty()){
                    int num=count(deadSem.begin(), deadSem.end(), essential_sem);
                    if(num>=7){
                        continue;
                    }
                }

                Rect r(xx,yy, w,h);
                if(display==true){
                    rectangle(ref_segrgb, r, Scalar(255, 255, 0), 2, LINE_AA);
                    imshow("reference image", ref_segrgb);
                    waitKey(0);
                }
                Mat temp_Mat = ref_Mat(r);
                Mat src(src_Mat.size(),CV_8UC1);
                src_Mat.convertTo(src, CV_8UC1);
                Mat temp(temp_Mat.size(),CV_8UC1);
                temp_Mat.convertTo(temp, CV_8UC1);
                Mat result = Mat::zeros(src.rows - temp.rows +1,src.cols - temp.cols +1, CV_32FC1);

                sortMatchedPoint(xVec, yVec, xRefVec, yRefVec, fineMatchedPosition);
                int yStart=0, xStart=0, yEnd=result.rows, xEnd=result.cols;
                //高度搜索范围
                for(size_t t=0; t<xVec.size(); t++){
                    //从上面的match点往下找，如果当前点比某个match点低,那它的高度搜索范围就是从上一个match点到当前match点
                    if(yy<yRefVec[t]){
                        //如果比第一个match点低，那高度搜索范围就是从顶边到第一个match点
                        if(t==0){
                            yStart=0;
                            yEnd=yVec[t];
                            break;
                        }
                        else{
                            yStart=yVec[t-1];
                            yEnd=yVec[t];
                            break;
                        }
                    }
                    //如果比最低match点还低，那就是最低match点到底边
                    else if(t==xVec.size()-1){
                        yStart=yVec[t];
                        yEnd=result.rows;
                        break;
                    }
                }
                //宽度搜索范围
                for(size_t t=0; t<xVec.size(); t++){
                    //从左往右找
                    if(xx<xRefVec[t]){
                        //如果比第一个match点左，那宽度搜索范围就是从左边到第一个match点
                        if(t==0){
                            xStart=0;
                            xEnd=xVec[t];
                            break;
                        }
                        else{
                            xStart=xVec[t-1];
                            xEnd=xVec[t];
                            break;
                        }
                    }
                    //如果比最右match点还右，那就是最右match点到右边
                    else if(t==xVec.size()-1){
                        xStart=xVec[t];
                        xEnd=result.cols;
                        break;
                    }
                }
                
                if(essential_sem==lastMajor_sem){
                    if(!coarseMatchedPosition.empty()){
                        vector<int> lastMatchedP = coarseMatchedPosition[coarseMatchedPosition.size()-1];
                        yStart=lastMatchedP[1];   
                    }   
                        
                }

                

                //calculate ConPro for each element of result matrix
                bool tempMatched =false;
                for (int y=yStart; y<yEnd; y+=3)
                {
                    for (int x=xStart; x<xEnd; x+=3)
                    {
                        vector<int> currentP{x,y};
                        // if(!coarseMatchedPosition.empty()){
                        //     Mat srcClone = src_segrgb.clone();
                        //     Rect rrr(x,y, w,h);
                        //     rectangle(srcClone, rrr, Scalar(255, 255, 255), 2, LINE_AA);
                        //     imshow("sliding window", srcClone);
                        //     waitKey(0);
                        // }
                        Mat src_part(src, cv::Rect(x,y,temp.cols, temp.rows));
                        set<int>major_sem = findMajorLabel(src_part, 0.2);
                        if(major_sem!=essential_sem){
                            continue;
                        }
                        bool overlap = nms(currentP,coarseMatchedPosition,w,h);
                        if(overlap==true){
                            continue;
                        }
                        ConPro = ConPro_calculator(src_part, temp, ConProMat);
                        if(ConPro<log(0.4)*temp.cols*temp.rows){
                            continue;
                        }

                        lastMajor_sem=major_sem;
                        // 先找到3个label的模板匹配，以这些label划出2个label的模板的搜索范围。
                        //然后以匹配到的2个label的模板，来划出下一个2个label的模板的搜索范围
                        vector<int>position{x,y,xx,yy};
                        bool fine=checkFineMatch(position, coarseMatchedPosition);

                        coarseMatchedPosition.push_back(position);
                        coarseMatchedNum+=1;
                        Rect r(x,y, w,h);
                        rectangle(src_segrgb, r, Scalar(255, 0, 0), 2, LINE_AA);
                        Rect rr(xx,yy, w,h);
                        rectangle(ref_segrgb, rr, Scalar(255, 0, 0), 2, LINE_AA);
                        
                        if(fine==true){
                            fineMatchedPosition.push_back(position);
                            rectangle(src_segrgb, r, Scalar(0, 0, 255), 2, LINE_AA);
                            rectangle(ref_segrgb, rr, Scalar(0, 0, 255), 2, LINE_AA);
                            fineMatchedNum+=1;
                        }

                        if(display==true){
                            imshow("source image", src_segrgb);
                            waitKey(0);
                        }
                        
                        tempMatched=true;
                        break;
                    }
                    if(tempMatched==true){
                        break;
                    }
                }
                if(tempMatched==false){
                    deadSem.push_back(essential_sem);
                    Rect rr(xx,yy, w,h);
                    rectangle(ref_segrgb, rr, Scalar(0, 0, 0), 2, LINE_AA);
                }
            }
            end=clock();
            t_diff=(double)(end-start)/CLOCKS_PER_SEC;
            cout<<"fineMatched: "<<fineMatchedNum<<",  "<<"coarseMatched: "<<coarseMatchedNum<<" time taken: "<<t_diff<<endl;
        }
        break;
    }
}