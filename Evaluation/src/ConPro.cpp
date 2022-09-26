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
            cout<<i<<": "<<s<<"     ";
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
    histMat = histMat.rowRange(0,20).colRange(0,20).clone();
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

    // cout<<left.size()<<","<<right.size()<<","<<up.size()<<","<<down.size()<<endl;
    
    // for(set<uchar>::iterator it=left.begin(); it!=left.end(); it++){
    //     cout<<"left: "<<int(*it)<<": "<<count(left_vec.begin(),left_vec.end(), *it)<<",  ";
    // }
    // cout<<endl;
    // for(set<uchar>::iterator it=right.begin(); it!=right.end(); it++){
    //     cout<<"right: "<<int(*it)<<": "<<count(right_vec.begin(),right_vec.end(), *it)<<",  ";
    // }
    // cout<<endl;
    // for(set<uchar>::iterator it=up.begin(); it!=up.end(); it++){
    //     cout<<"up: "<<int(*it)<<": "<<count(up_vec.begin(),up_vec.end(), *it)<<",  ";
    // }
    // cout<<endl;
    // for(set<uchar>::iterator it=down.begin(); it!=down.end(); it++){
    //     cout<<"down: "<<int(*it)<<": "<<count(down_vec.begin(),down_vec.end(), *it)<<",  ";
    // }
    // cout<<endl;

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
                    xy.push_back(0);//this is a bool value to see if this template has been matched
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
                bool badPattern = checkTempPattern(temp_Mat);
                if(badPattern==true){
                    continue;
                }
                // bool repeatPattern = checkRepeatPattern(major_sem, fineTemp);
                // if(repeatPattern==true){
                //     continue;
                // }
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

//find a group of templates that have the same labels as input template and are closest to input template
bool findGroup(vector<vector<int>>& tempGroup, vector<vector<int>> allTemplates, vector<int>targetLabels){
    bool groupFound=false; //if a group of templates are found
    vector<int> mostClose; //the template that's closet to input template
    vector<int> secondClose; //the template that's second close to input template
    float minD, minD2; //minD distance between mostClose and input template, minD2 distance btw secondClose and input template
    size_t inputIdx = tempGroup[0][2]; //index of input template
    vector<int>inputP{tempGroup[0][0], tempGroup[0][1]}; //x,y of input template
    for(size_t st=0; st<allTemplates.size(); st++){
        //if the current template is the input template, move on
        if(st==inputIdx){
            continue;
        }
        vector<int>Labels{allTemplates[st][2],allTemplates[st][3]}; //labels of current template
        //if current template label is different from the input template, move on
        if(Labels!=targetLabels){
            continue;
        }
        vector<int>currentP{allTemplates[st][0],allTemplates[st][1]}; //x,y of current template
        float distance = norm(inputP, currentP);
        //if this is the first template, mark it as mostClose
        if(mostClose.empty()){
            mostClose=currentP;
            mostClose.push_back(int(st));
            mostClose.push_back(allTemplates[st][4]);
            minD=distance;
        }
        else if(distance<minD){
            secondClose=mostClose;
            minD2=minD;
            mostClose=currentP;
            mostClose.push_back(int(st));
            mostClose.push_back(allTemplates[st][4]);
            minD=distance;
        }
        else if(secondClose.empty()){
            secondClose=currentP;
            secondClose.push_back(int(st));
            secondClose.push_back(allTemplates[st][4]);
            minD2=distance;
        }
        else if(distance<minD2){
            secondClose=currentP;
            secondClose.push_back(int(st));
            secondClose.push_back(allTemplates[st][4]);
            minD2=distance;
        }
    }
    if(!mostClose.empty() && !secondClose.empty()){
        groupFound=true;
        tempGroup.push_back(mostClose);
        tempGroup.push_back(secondClose);
    }
    
    return groupFound;
}


bool matchGroup(vector<vector<int>>qryGroup, vector<vector<int>>refGroup){
    bool matched=false;
    float thre=0.1;
    int qryF1 = qryGroup[0][3];
    int qryF2 = qryGroup[1][3];
    int qryF3 = qryGroup[2][3];
    int refF1 = refGroup[0][3];
    int refF2 = refGroup[1][3];
    int refF3 = refGroup[2][3];
    if(qryF1!=refF1 || qryF2!=refF2 || qryF3!=refF3){
        return matched;
    }

    vector<int>qryP1{qryGroup[0][0],qryGroup[0][1]};
    vector<int>qryP2{qryGroup[1][0],qryGroup[1][1]};
    vector<int>qryP3{qryGroup[2][0],qryGroup[2][1]};
    vector<int>refP1{refGroup[0][0],refGroup[0][1]};
    vector<int>refP2{refGroup[1][0],refGroup[1][1]};
    vector<int>refP3{refGroup[2][0],refGroup[2][1]};
    float qryD1=norm(qryP1, qryP2);
    float qryD2=norm(qryP1, qryP3);
    float qryD3=norm(qryP2, qryP3);
    float refD1=norm(refP1, refP2);
    float refD2=norm(refP1, refP3);
    float refD3=norm(refP2, refP3);
    float qryR1=qryD1/qryD2;
    float qryR2=qryD1/qryD3;
    float qryR3=qryD2/qryD3;
    float refR1=refD1/refD2;
    float refR2=refD1/refD3;
    float refR3=refD2/refD3;
    float diffQR1=abs(qryR1-refR1)/refR1;
    float diffQR2=abs(qryR2-refR2)/refR2;
    float diffQR3=abs(qryR3-refR3)/refR3;

    if(diffQR1<thre && diffQR2<thre && diffQR3<thre){
        matched=true;
    }


    return matched;
}

int Match(vector<vector<int>> qryTemplates, vector<vector<int>> refTemplates, Mat qryImg_rgb, Mat refImg_rgb, int w, int h, bool display){
    int matches=0;
    for(size_t st=0; st<qryTemplates.size(); st++){
        Mat qryImg = qryImg_rgb.clone();
        Mat refImg = refImg_rgb.clone();
        vector<int> qryTemplate=qryTemplates[st];
        //if this template has been matched, move on
        if(qryTemplate[4]==1){
            continue;
        }
        vector<int> qryLabels{qryTemplate[2], qryTemplate[3]};//semantic labels of current template
        //find a group of templates close to the current template
        vector<vector<int>>qryGroup;//a group of close query templates, [[x,y,idx,flag],[x,y,idx,flag],[x,y,idx, flag]]
        vector<int> xyIdx{qryTemplate[0],qryTemplate[1],int(st), qryTemplate[4]};
        qryGroup.push_back(xyIdx);
        bool groupFound = findGroup(qryGroup, qryTemplates, qryLabels);
        //if we cannot find a group for this query template, move on
        if(groupFound==false){
            continue;
        }

        for(size_t r=0; r<refTemplates.size(); r++){
            vector<int>refTemplate=refTemplates[r];
            //if this template has been matched, move on
            if(refTemplate[4]==1){
                continue;
            }
            //if the reference template label is not the same as the query template label, move on
            vector<int>refLabels{refTemplate[2],refTemplate[3]};
            if(refLabels!=qryLabels){
                continue;
            }
            //find a group of templates close to the current template
            vector<vector<int>>refGroup;//a group of close ref templates, [[x,y,idx],[x,y,idx],[x,y,idx]]
            vector<int> xyIdx{refTemplate[0],refTemplate[1],int(r), refTemplate[4]};
            refGroup.push_back(xyIdx);
            bool groupFound = findGroup(refGroup, refTemplates, refLabels);
            //if we cannot find a group for this reference template, move on
            if(groupFound==false){
                continue;
            }
            //match the refGroup and qryGroup
            bool groupMatched = matchGroup(qryGroup, refGroup);
            //if the groups are matched, turn the flag of all templates in both groups to 1
            if(groupMatched==true){
                // cout<<"group is matched"<<endl;
                for(size_t tt=0; tt<qryGroup.size(); tt++){
                    int idx = qryGroup[tt][2];
                    qryTemplates[idx][4]=1;
                    Rect r(qryTemplates[idx][0],qryTemplates[idx][1], w, h);
                    rectangle(qryImg, r, Scalar(0, 0, 255), 2, LINE_AA); 
                }
                for(size_t tt=0; tt<refGroup.size(); tt++){
                    int idx = refGroup[tt][2];
                    refTemplates[idx][4]=1;
                    Rect r(refTemplates[idx][0],refTemplates[idx][1], w, h);
                    rectangle(refImg, r, Scalar(0, 0, 255), 2, LINE_AA);     
                }
                // cout<<"this is reached"<<endl;
                if(display==true){
                    imshow("query image", qryImg);
                    imshow("refrence image", refImg);
                    waitKey(0);
                }
                break;
            }
        }
    }
    //count the number of templates matched in query templates
    for(size_t st=0; st<qryTemplates.size(); st++){
        if(qryTemplates[st][4]==1){
            matches+=1;
        }
    }
    
    for(size_t st=0; st<refTemplates.size(); st++){
        if(refTemplates[st][4]==1){
            matches+=1;
        }
    }



    return matches;
}





int main(int argc, char **argv)
{
    bool display=false;
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
    string seg_path = "/media/jialin/045E58135E57FC3C/UBUNTU/Evaluation_segmented/overcast-summer/rear";
    // string seg_path=kitti360+"data_2d_raw/2013_05_28_drive_0010_sync/image_00/segmentation";
    vector<cv::String> fn_seg;
    glob(seg_path, fn_seg, false);


    string seg_path2 = "/media/jialin/045E58135E57FC3C/UBUNTU/Evaluation_segmented/snow/rear";
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


    string segrgb_path2 = "/media/jialin/045E58135E57FC3C/UBUNTU/Evaluation_segmented/snow/rear/seg-rgb";
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
        
        vector<vector<int>> refTemplates;
        vector<vector<int>> fineTemplates;
        bool templateOkay = templateFinder(refTemplates, fineTemplates,ref_segrgb, ref_Mat, w, h, ignoreLabels);
        end=clock();
        t_diff=(double)(end-start)/CLOCKS_PER_SEC;
        cout<<"time to find template: "<<t_diff<<endl;
        if(templateOkay==false){
            cout<<"image"<<to_string(i)<<"doesn't have a good template"<<endl;
            imgSkip+=1;
            cout<<"skipped img number: "<<imgSkip<<endl;
            continue;
        }
        cout<<"ref template found: "<<refTemplates.size()<<endl;
        // exit(0);


        for(int ii=1; ii<8; ii++){
            start=clock();
            Mat src_Mat = imread(fn_seg2[ii+2], IMREAD_GRAYSCALE);
            Mat src_segrgb = imread(fn_segrgb2[ii+2], IMREAD_COLOR);

            if (src_Mat.empty() || src_segrgb.empty())
            {
                cout << "failed to read image" << endl;
                return EXIT_FAILURE;
            }

            vector<vector<int>> qryTemplates;
            vector<vector<int>> fineTemplates2;
            bool templateOkay = templateFinder(qryTemplates, fineTemplates2,src_segrgb, src_Mat, w, h, ignoreLabels);
            // end=clock();
            // t_diff=(double)(end-start)/CLOCKS_PER_SEC;
            // cout<<"time to find template: "<<t_diff<<endl;
            if(templateOkay==false){
                cout<<"image"<<to_string(i)<<"doesn't have a good template"<<endl;
                imgSkip+=1;
                cout<<"skipped img number: "<<imgSkip<<endl;
                continue;
            }
            // cout<<"qry template found: "<<qryTemplates.size()<<endl;

            Mat refClone = ref_segrgb.clone();
            Mat qryClone = src_segrgb.clone();
            int templateMatched = Match(qryTemplates, refTemplates, qryClone, refClone, w, h, display);
            float matchedPortion=float(templateMatched)/float(qryTemplates.size()+refTemplates.size());
            end=clock();
            t_diff=(double)(end-start)/CLOCKS_PER_SEC;
            cout<<"templates matched: "<<templateMatched<<"     portion:"<<matchedPortion<<"     time cost:"<<t_diff<<endl;

        }
        break;
    }
}