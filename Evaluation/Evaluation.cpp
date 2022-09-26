#include <iostream>
#include <fstream>
#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/imgcodecs.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <set>
#include <ctime>
#include <numeric>
#include "imgCompare.h"
#include "templateExtractor.h"

using namespace std;
using namespace cv;

//path to kitti360 dataset
string semanticImgPath = "/media/jialin/045E58135E57FC3C/UBUNTU/Evaluation_segmented/";
string rawImgPath = "/media/jialin/045E58135E57FC3C/UBUNTU/Evaluation/";

vector<string> loadLocationImgPath(string locationIdx, string condition, string datasetPath){
    vector<string> outputVec;
    ifstream inFile;
    string folderPath = datasetPath + "queries_per_location";
    vector<cv::String> fn;
    glob(folderPath, fn, false);

    for(int i=0; i<fn.size(); i++){
        //find the target txt file
        string filePath = fn[i];
        string::size_type idx = filePath.find(locationIdx+".txt", 0);
        if(idx==string::npos){
            continue;
        }
        // cout<<"current location: "<<locationIdx<<endl;

        //open txt file
        inFile.open(filePath);
        if(!inFile){
            cout<<"unable to open file: "<<filePath <<endl;
            exit(1);
        }

        //read the txt file
        while(!inFile.eof()){
            string inLine;
            getline(inFile, inLine,'\n');
            idx=inLine.rfind(condition+"/rear",0);
            if(idx!=string::npos){
                if(datasetPath==semanticImgPath){
                    int strLen = inLine.length();
                    string pngName = inLine.substr(0, strLen-4);
                    pngName = pngName + ".png";
                    outputVec.push_back(datasetPath + pngName);

                }
                else{
                    outputVec.push_back(datasetPath + inLine);
                }
                
            }
        }

        //close the file
        inFile.close();
    }

    return outputVec;
}

void readImg(vector<string> imgPath, vector<Mat>& imgVec){
    imgVec.clear();
    imgVec.reserve(imgPath.size());
    for(size_t st=0; st<imgPath.size(); st++){
        Mat imgMat = imread(imgPath[st], IMREAD_GRAYSCALE);
        imgVec.push_back(imgMat);
    }
}

vector<string> getLocationNum(int start, int stop){
    vector<string> output;
    for(int i=start; i<stop; i++){
        string num=to_string(i);
        while (num.length()<3)
        {
            num="0"+num;
        }
        output.push_back(num);
        // cout<<num<<endl;
    }
    return output;
}

struct imgS
{
    string location;
    string condition;
    vector<vector<int>> fineT;
    vector<vector<int>> coarseT;

};

float getMinScore(vector<struct imgS> inputV){
    float minScore=1.0;
    struct imgS cFrame = inputV[inputV.size()-1];
    vector<vector<int>> rFineT, rCoarseT;
    rFineT==cFrame.fineT;
    rCoarseT=cFrame.coarseT;
    for(size_t st=0; st<inputV.size()-1; st++){
        vector<vector<int>> qFineT, qCoarseT;
        qFineT=inputV[st].fineT;
        qCoarseT=inputV[st].coarseT;
        float score = imgCompare(rFineT, rCoarseT, qFineT, qCoarseT);
        if(score<minScore){
            minScore=score;
        }
    }
    return minScore;
}

int main(){
    int w=15, h=15;
    vector<int> ignoreLabels = get_ignoreLabels();
    clock_t start, end; //timing
    double t_diff;
    vector<string> conditionVec{"rain", "night", "night-rain", "dawn", "dusk", "overcast-summer", "overcast-winter","snow","sun"};
    // vector<string> conditionVec{"night"};
    vector<string> locationNumVec = getLocationNum(15,25);
    vector<vector<vector<struct imgS>>> vvvsLocationImgs;
    for(size_t con=0; con<conditionVec.size(); con++){
        start=clock();
        string refCondition=conditionVec[con];
        vector<vector<struct imgS>> vvsLocationImgs;
        cout<<"extracting "+refCondition+" templates...."<<endl;
        for(size_t st=0; st<locationNumVec.size(); st++){
            string refLocation=locationNumVec[st];
            vector<string> locationImgPath = loadLocationImgPath(refLocation, refCondition, semanticImgPath);
            if(locationImgPath.empty()){
                continue;
            }
            //read images from the image path
            vector<Mat> imgVec;
            readImg(locationImgPath, imgVec);
            // cout<<"144 reached"<<endl;
            //build structure
            vector<struct imgS> imgSVec;
            for(size_t st=0; st<imgVec.size(); st++){
                struct imgS img;
                img.location=refLocation;
                img.condition=refCondition;
                vector<vector<int>> fineT, coarseT;
                bool templateFound = templateExtractor(coarseT, fineT, imgVec[st], w, h, ignoreLabels);
                img.fineT=fineT;
                img.coarseT=coarseT;
                imgSVec.push_back(img);
            }
            vvsLocationImgs.push_back(imgSVec);
        }
        vvvsLocationImgs.push_back(vvsLocationImgs);
        end=clock();
        t_diff=(double)(end-start)/CLOCKS_PER_SEC;
        cout<<"time taken : "<<t_diff<<endl;

    }
    
    cout<<"comparing ..."<<endl;
    for(size_t con=0; con<conditionVec.size(); con++){
        start=clock();
        string refCondition = conditionVec[con];
        cout<<"refCondition: "<<refCondition<<endl;
        vector<vector<struct imgS>> refLocationImgs=vvvsLocationImgs[con];
        if(refLocationImgs[0][0].condition!=refCondition){
            cout<<"wrong condition!!!!!!!!!!!!!!!"<<endl;
            exit(0);
        }
        start=clock();

        for(size_t qCon=0; qCon<conditionVec.size(); qCon++){
            int candidateCount=0;
            int matchCount=0;
            int groundTrueCount=0;
            vector<vector<struct imgS>> qryLocationImgs=vvvsLocationImgs[qCon];
            string qryCondition = conditionVec[qCon];
            cout<<"     qryCondition: "<<qryCondition<<endl;
            if(qryLocationImgs[0][0].condition!=qryCondition){
                cout<<"wrong condition!!!!!!!!!!!!!!!"<<endl;
                exit(0);
            }
            // cout<<"start comparing ..."<<endl;
            for(size_t r=0; r<refLocationImgs.size(); r++){
                vector<struct imgS> refL = refLocationImgs[r];
                struct imgS cFrame = refL[refL.size()-1];
                string refLocation = cFrame.location;
                vector<vector<int>> rFineT, rCoarseT;
                rFineT==cFrame.fineT;
                rCoarseT=cFrame.coarseT;
                float minScore=getMinScore(refL);
                // cout<<"ref: "<<int(r)<<"minScore: "<<minScore<<endl;

                vector<string> primaryCandidates;
                vector<float> groupScoreVec;
                float maxScore=0.;
                for(size_t q=0; q<qryLocationImgs.size(); q++){
                    vector<struct imgS> qryL = qryLocationImgs[q];
                    float groupScore=0.;
                    bool qryOkay=false;
                    string qryLocation = qryL[0].location;
                    if(qryLocation==refLocation){
                        groundTrueCount+=1;
                    }
                    for(size_t st=0; st<qryL.size(); st++){
                        vector<vector<int>> qFineT, qCoarseT;
                        qFineT=qryL[st].fineT;
                        qCoarseT=qryL[st].coarseT;
                        float score = imgCompare(rFineT, rCoarseT, qFineT, qCoarseT);
                        // cout<<"     "<<int(st)<<":    "<<score<<endl;
                        if(score>minScore){
                            qryOkay=true;
                        }
                        groupScore+=score;
                    }
                    if(qryOkay==true){
                        primaryCandidates.push_back(qryLocation);
                        float avgScore=groupScore/float(qryL.size());
                        groupScoreVec.push_back(avgScore);
                        if(avgScore>maxScore){
                            maxScore=avgScore;
                        }
                    }
                }

                vector<string> finalCandidates;
                float gsThreshold = maxScore*0.75;
                for(size_t st=0; st<primaryCandidates.size(); st++){
                    if(groupScoreVec[st]>gsThreshold){
                        finalCandidates.push_back(primaryCandidates[st]);
                    }
                }

                if(find(finalCandidates.begin(), finalCandidates.end(), refLocation)!=finalCandidates.end()){
                    matchCount+=1;
                    // cout<<"match found"<<endl;
                }
                candidateCount+=finalCandidates.size();
                // cout<<"candidates: "<<finalCandidates.size()<<endl;
            }

            float accuracyRate = float(matchCount)/float(candidateCount);
            float recallRate = float(matchCount)/float(groundTrueCount);
            // cout<<"match count: "<<matchCount<<endl;
            // cout<<"candidate count: "<<candidateCount<<endl;
            // cout<<"groundTrueCount: "<<groundTrueCount<<endl;
            cout<<"     accuracy:  "<<accuracyRate<<",     "<<"recall rate: "<<recallRate<<endl;
        }
        end=clock();
        t_diff=(double)(end-start)/CLOCKS_PER_SEC;
        cout<<"time taken : "<<t_diff<<endl;
    }

    return 0;
}