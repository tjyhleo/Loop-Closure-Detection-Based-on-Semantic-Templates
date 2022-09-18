#include "Thirdparty/DBoW2/DBoW2/BowVector.h"
#include "Thirdparty/DBoW2/DBoW2/FeatureVector.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include "include/ORBextractor.h"
#include "include/ORBVocabulary.h"
#include <iostream>
#include <vector>
#include <string>

using namespace cv;
using namespace std;

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

/**
 * @brief 提取图像的ORB特征点，提取的关键点存放在mvKeys，描述子存放在mDescriptors
 * 
 * @param[in] flag          标记是左图还是右图。0：左图  1：右图
 * @param[in] im            等待提取特征点的图像
 */
void ExtractORB(const vector<cv::Mat> vmImg, vector<cv::Mat>& vmDescriptors)
{
    // ORB_SLAM2::ORBextractor *mpORBextractorLeft = new ORB_SLAM2::ORBextractor(); 
    
    // ORB_SLAM2::ORBextractor asdf();
    int nFeatures =1000;
    float fScaleFactor=1.2;
    int nLevels=8;
    int fIniThFAST=20;
    int fMinThFAST=8;
    ORB_SLAM2::ORBextractor* mpORBextractor = new ORB_SLAM2::ORBextractor(2*nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);
    cv::Mat img;
    cv::Mat mDescriptors;
    std::vector<cv::KeyPoint> mvKeys;
    for(size_t i=0; i<vmImg.size(); i++){
        img = vmImg[i];
        // 左图的话就套使用左图指定的特征点提取器，并将提取结果保存到对应的变量中 
        // 这里使用了仿函数来完成，重载了括号运算符 ORBextractor::operator() 
        (*mpORBextractor)(img,				//待提取特征点的图像
                            cv::Mat(),		//掩摸图像, 实际没有用到
                            mvKeys,			//输出变量，用于保存提取后的特征点
                            mDescriptors);	//输出变量，用于保存特征点的描述子
        vmDescriptors.push_back(mDescriptors);
    }
    delete mpORBextractor;
    
    
}

DBoW2::BowVector ComputeBoW(cv::Mat mDescriptors)
{
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


void ReadVocabulary(ORB_SLAM2::ORBVocabulary *voc){
    //Load ORB Vocabulary
    cout << endl << "Loading ORB Vocabulary. This could take a while..." << endl;
    //获取字典加载状态
    string strVocFile = "Vocabulary/ORBvoc.txt";
    bool bVocLoad = voc->loadFromTextFile(strVocFile);
    //如果加载失败，就输出调试信息
    if(!bVocLoad)
    {
        cerr << "Wrong path to vocabulary. " << endl;
        cerr << "Falied to open at: " << strVocFile << endl;
        //然后退出
        exit(-1);
    }
    //否则则说明加载成功
    cout << "Vocabulary loaded!" << endl << endl;
}

void ReadImg(vector<cv::Mat> &vmImages, cv::String img_path){
    vector<cv::String> fn_img;
    glob(img_path, fn_img, true);
    for(size_t i=0; i<fn_img.size();i++){
        vmImages.push_back(imread(fn_img[i],IMREAD_GRAYSCALE));
        // cv::imshow("asdf",vmImages[i]);
        // waitKey(0);
    }
    
    
}

void BuildVoc(ORB_SLAM2::ORBVocabulary *voc, vector<cv::Mat>vmDescriptors){
    vector<vector<cv::Mat>> vvmDescriptors;
    for(size_t i=0; i<vmDescriptors.size(); i++){
        vector<cv::Mat> vCurrentDesc = toDescriptorVector(vmDescriptors[i]);
        vvmDescriptors.push_back(vCurrentDesc);
    }
    voc ->create(vvmDescriptors);
    
}


bool DetectLoop(vector<cv::Mat> refDes, vector<cv::Mat> queryDes){
    cv::Mat currentDes = refDes[refDes.size()-1];
    DBoW2::BowVector cBowVec = ComputeBoW(currentDes);
    float minScore=1.;
    bool detected = false;
    for(size_t i=0; i<refDes.size()-1; i+=5){
        DBoW2::BowVector BowVec = ComputeBoW(refDes[i]);
        float s = mpVocabulary->score(cBowVec, BowVec);
        cout<<i<<": "<<s<<endl;
        if(s<minScore){
            minScore = s;
        }
    }

    cout<<"minScore: "<<minScore<<endl;

    for(size_t i=80; i<queryDes.size(); i++){
        DBoW2::BowVector BowVec = ComputeBoW(queryDes[i]);
        float s = mpVocabulary->score(cBowVec, BowVec);
        cout<<"s: "<<s<<endl;
        if(s>=minScore){
            detected=true;
            // break;
        }
    }

    return detected;

}

int main(){

    //读取图片
    cout<<"reading images..."<<endl;
    vector<cv::Mat> vmRefImages;
    cv::String refImg_path = "/media/jialin/045E58135E57FC3C/UBUNTU/Evaluation_data/overcast/scene2/*.png";
    ReadImg(vmRefImages, refImg_path);
    vector<cv::Mat> vmQryImages;
    cv::String qryImg_path = "/media/jialin/045E58135E57FC3C/UBUNTU/Evaluation_data/night/scene2/*.png";
    ReadImg(vmQryImages, qryImg_path);

    //提取ORB descripter
    cout<<"extracting ORB descriptors..."<<endl;
    vector<cv::Mat> vmRefDescriptors;
    ExtractORB(vmRefImages, vmRefDescriptors);
    vector<cv::Mat> vmQryDescriptors;
    ExtractORB(vmQryImages, vmQryDescriptors);
    
    //加载字典
    cout<<"loading vocabulary..."<<endl;
    // string f_path = "/home/jialin/Documents/VSC_Projects/BoW_ORB/Vocabulary/vocab_Oxford.yml.gz";
    // mpVocabulary->load(f_path);
    string f_path = "/home/jialin/Documents/VSC_Projects/BoW_ORB/Vocabulary/vocab_Oxford.txt";
    mpVocabulary -> loadFromTextFile(f_path);
    cout<<*mpVocabulary<<endl;

    //检测回环
    cout<<"detecting loop closure candidate frames..."<<endl;
    bool loop_detected = DetectLoop(vmRefDescriptors, vmQryDescriptors);
    if(loop_detected){
        cout<<"loop detected"<<endl;
    }
    else{
        cout<<"loop closure not detected"<<endl;
    }

    

    delete mpVocabulary;
    return 0;
}




