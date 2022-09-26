#include "templateExtractor.h"

void get_histMat(Mat &histMat){
    FileStorage fs_read;
    double val_min, val_max;
    fs_read.open("histMat_unS.xml", FileStorage::READ);
    if(!fs_read.isOpened()){
        cout<<"histMat_unS.xml is not opened"<<endl;
        exit(1);
    }
    fs_read["histMat"]>>histMat;
    fs_read.release();
    histMat = histMat.rowRange(0,20).colRange(0,20).clone();
    // minMaxLoc(histMat, &val_min, &val_max, NULL, NULL);
    // cout<<"histMat min, max: "<<val_min<<", "<<val_max<<endl;

    //get sums of histMat
    vector<float> row_sum;
    for(int i=0; i<histMat.rows; i++){
        row_sum.push_back(sum(histMat.row(i))[0]);
    }

    //convert histMat into a matrix of conditional probability
    histMat.convertTo(histMat,CV_32FC1);
    for(int i=0; i<histMat.rows; i++){
        histMat.row(i) = histMat.row(i) / row_sum[i];
    }
}

vector<int> get_ignoreLabels(){
    Mat ConProMat;
    get_histMat(ConProMat);
    vector<int> ignoreLabels;
    for(int i = 0; i<ConProMat.rows; i++){
        float s = ConProMat.at<float>(i,i);
        if(s<0.6){
            ignoreLabels.push_back(i);
        }
    }
    // int dynamic[]{24,25,26,27,28,29,30,31,32,33};
    // ignoreLabels.insert(ignoreLabels.end(),dynamic,dynamic+10);
    // int terrain =22;
    int dynamic[]{11,12,13,14,15,16,17,18};
    ignoreLabels.insert(ignoreLabels.end(),dynamic,dynamic+8);
    int terrain =9;
    ignoreLabels.push_back(terrain);
    return ignoreLabels;
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
            }
        }
        if(counter>1){
            bad_pattern=true;
        }
    }

    if(bad_pattern==true){
        return bad_pattern;
    }

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

bool templateExtractor(vector<vector<int>>& coarseTemp,vector<vector<int>>& fineTemp, 
                    Mat ref_Mat, const int w, const int h, const vector<int> ignoreLabels){
    bool templateOkay = false;
    bool lowerRequire=false;
    int imgH = ref_Mat.rows;
    int imgW = ref_Mat.cols;
    
    for(int j=0; j<imgH - h; j+=3){
        for (int i=0; i<imgW - w; i+=3){
            Rect r(i,j, w,h);

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
                if (num > w*h*0.2){
                    counter+=1;
                    major_sem.insert(*it);
                }
            }
            if (counter<2){
                continue;
            }

            else if(counter==2){
                //ensure each type of label occupy more than 40%
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
                    // rectangle(img, r, Scalar(255, 255, 255), 1, LINE_AA);
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
                xy.push_back(0);
                for(set<uchar>::iterator it=major_sem.begin(); it!=major_sem.end(); it++){
                    xy.push_back(int(*it));
                }
                fineTemp.push_back(xy);
                // rectangle(img, r, Scalar(0, 0, 255), 1, LINE_AA);
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


