#include "Thirdparty/DBoW2/DBoW2/BowVector.h"
#include "Thirdparty/DBoW2/DBoW2/FeatureVector.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include "include/ORBextractor.h"
#include "include/ORBVocabulary.h"
#include <iostream>
#include <fstream>
#include <set>
#include <ctime>
#include <numeric>


using namespace std;
using namespace cv;

//path to kitti360 dataset
string semanticImgPath = "/media/jialin/045E58135E57FC3C/UBUNTU/Evaluation_segmented/";
string rawImgPath = "/media/jialin/045E58135E57FC3C/UBUNTU/Evaluation/";
ORB_SLAM2::ORBVocabulary* mpVocabulary = new ORB_SLAM2::ORBVocabulary();

std::vector<cv::Mat> toDescriptorVector(const cv::Mat &Descriptors)
{
	//存储转换结果的向量
    std::vector<cv::Mat> vDesc;
	//创建保留空间
    vDesc.reserve(Descriptors.rows);
	//对于每一个特征点的描述子
    for (int j=0;j<Descriptors.rows;j++)
		//从描述子这个矩阵中抽取出来存到向量中
        vDesc.push_back(Descriptors.row(j));
	
	//返回转换结果
    return vDesc;
}

void ExtractORB(const Mat img, Mat & mDescriptors)
{
    // ORB_SLAM2::ORBextractor *mpORBextractorLeft = new ORB_SLAM2::ORBextractor(); 
    
    // ORB_SLAM2::ORBextractor asdf();
    int nFeatures =1000;
    float fScaleFactor=1.2;
    int nLevels=8;
    int fIniThFAST=20;
    int fMinThFAST=8;
    ORB_SLAM2::ORBextractor* mpORBextractor = new ORB_SLAM2::ORBextractor(2*nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);
    std::vector<cv::KeyPoint> mvKeys;
    (*mpORBextractor)(img,				//待提取特征点的图像
                    cv::Mat(),		//掩摸图像, 实际没有用到
                    mvKeys,			//输出变量，用于保存提取后的特征点
                    mDescriptors);	//输出变量，用于保存特征点的描述子
    delete mpORBextractor;
    
}

DBoW2::BowVector ComputeBoW(cv::Mat img)
{

    cv::Mat mDescriptors;
    ExtractORB(img, mDescriptors);
    DBoW2::BowVector mBowVec;
    DBoW2::FeatureVector mFeatVec;
    // 将描述子mDescriptors转换为DBOW要求的输入格式
    vector<cv::Mat> vCurrentDesc = toDescriptorVector(mDescriptors);
    // 将特征点的描述子转换成词袋向量mBowVec以及特征向量mFeatVec
    mpVocabulary->transform(vCurrentDesc,	//当前的描述子vector
                            mBowVec,			//输出，词袋向量，记录的是单词的id及其对应权重TF-IDF值
                            mFeatVec,		//输出，记录node id及其对应的图像 feature对应的索引
                            4);				//4表示从叶节点向前数的层数
    return mBowVec;
}




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
    DBoW2::BowVector cBowVec;

};

float getMinScore(vector<struct imgS> inputV){
    float minScore=1.0;
    struct imgS cFrame = inputV[inputV.size()-1];
    DBoW2::BowVector rBowVec = cFrame.cBowVec;
    DBoW2::BowVector qBowVec;
    for(size_t st=0; st<inputV.size()-1; st++){
        qBowVec= inputV[st].cBowVec;
        float score= mpVocabulary->score(rBowVec, qBowVec);
        if(score<minScore){
            minScore=score;
        }
    }
    return minScore;
}

int main(){
    //加载字典
    cout<<"loading vocabulary..."<<endl;
    string f_path = "/home/jialin/Documents/VSC_Projects/Evaluation/Vocabulary/vocab_Oxford.txt";
    mpVocabulary -> loadFromTextFile(f_path);
    cout<<*mpVocabulary<<endl;
    clock_t start, end; //timing
    double t_diff;
    vector<string> conditionVec{"rain", "night", "night-rain", "dawn", "dusk", "overcast-summer", "overcast-winter","snow","sun"};
    // vector<string> conditionVec{"rain"};
    // vector<string> locationNumVec = getLocationNum(15,25);
    vector<string> locationNumVec = getLocationNum(15,25);
    vector<vector<vector<struct imgS>>> vvvsLocationImgs;
    for(size_t con=0; con<conditionVec.size(); con++){
        start=clock();
        string refCondition=conditionVec[con];
        vector<vector<struct imgS>> vvsLocationImgs;
        cout<<"extracting "+refCondition+" descriptors...."<<endl;
        for(size_t st=0; st<locationNumVec.size(); st++){
            string refLocation=locationNumVec[st];
            vector<string> locationImgPath = loadLocationImgPath(refLocation, refCondition, rawImgPath);
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
                DBoW2::BowVector cBowVec = ComputeBoW(imgVec[st]);
                img.cBowVec = cBowVec;
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
                DBoW2::BowVector rBowVec = cFrame.cBowVec; 
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
                        DBoW2::BowVector qBowVec = qryL[st].cBowVec;
                        float score = mpVocabulary->score(rBowVec, qBowVec);
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