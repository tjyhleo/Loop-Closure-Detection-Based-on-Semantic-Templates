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
// vector<KeyFrame*> KeyFrameDatabase::DetectLoopCandidates(KeyFrame* pKF, float minScore)
// {
//     // 取出与当前关键帧相连（>15个共视地图点）的所有关键帧，这些相连关键帧都是局部相连，在闭环检测的时候将被剔除
//     // 相连关键帧定义见 KeyFrame::UpdateConnections()
//     set<KeyFrame*> spConnectedKeyFrames = pKF->GetConnectedKeyFrames();

//     // 用于保存可能与当前关键帧形成闭环的候选帧（只要有相同的word，且不属于局部相连（共视）帧）
//     list<KeyFrame*> lKFsSharingWords;

//     // Search all keyframes that share a word with current keyframes
//     // Discard keyframes connected to the query keyframe
//     // Step 1：找出和当前帧具有公共单词的所有关键帧，不包括与当前帧连接（也就是共视）的关键帧
//     {
//         unique_lock<mutex> lock(mMutex);

//         // words是检测图像是否匹配的枢纽，遍历该pKF的每一个word
//         // mBowVec 内部实际存储的是std::map<WordId, WordValue>
//         // WordId 和 WordValue 表示Word在叶子中的id 和权重
//         for(DBoW2::BowVector::const_iterator vit=pKF->mBowVec.begin(), vend=pKF->mBowVec.end(); vit != vend; vit++)
//         {
//             // 提取所有包含该word的KeyFrame
//             list<KeyFrame*> &lKFs =   mvInvertedFile[vit->first];
//             // 然后对这些关键帧展开遍历
//             for(list<KeyFrame*>::iterator lit=lKFs.begin(), lend= lKFs.end(); lit!=lend; lit++)
//             {
//                 KeyFrame* pKFi=*lit;
                
//                 if(pKFi->mnLoopQuery!=pKF->mnId)    
//                 {
//                     // 还没有标记为pKF的闭环候选帧
//                     pKFi->mnLoopWords=0;
//                     // 和当前关键帧共视的话不作为闭环候选帧
//                     if(!spConnectedKeyFrames.count(pKFi))
//                     {
//                         // 没有共视就标记作为闭环候选关键帧，放到lKFsSharingWords里
//                         pKFi->mnLoopQuery=pKF->mnId;
//                         lKFsSharingWords.push_back(pKFi);
//                     }
//                 }
//                 pKFi->mnLoopWords++;// 记录pKFi与pKF具有相同word的个数
//             }
//         }
//     }

//     // 如果没有关键帧和这个关键帧具有相同的单词,那么就返回空
//     if(lKFsSharingWords.empty())
//         return vector<KeyFrame*>();

//     list<pair<float,KeyFrame*> > lScoreAndMatch;

//     // Only compare against those keyframes that share enough words
//     // Step 2：统计上述所有闭环候选帧中与当前帧具有共同单词最多的单词数，用来决定相对阈值 
//     int maxCommonWords=0;
//     for(list<KeyFrame*>::iterator lit=lKFsSharingWords.begin(), lend= lKFsSharingWords.end(); lit!=lend; lit++)
//     {
//         if((*lit)->mnLoopWords>maxCommonWords)
//             maxCommonWords=(*lit)->mnLoopWords;
//     }

//     // 确定最小公共单词数为最大公共单词数目的0.8倍
//     int minCommonWords = maxCommonWords*0.8f;

//     int nscores=0;

//     // Compute similarity score. Retain the matches whose score is higher than minScore
//     // Step 3：遍历上述所有闭环候选帧，挑选出共有单词数大于minCommonWords且单词匹配度大于minScore存入lScoreAndMatch
//     for(list<KeyFrame*>::iterator lit=lKFsSharingWords.begin(), lend= lKFsSharingWords.end(); lit!=lend; lit++)
//     {
//         KeyFrame* pKFi = *lit;

//         // pKF只和具有共同单词较多（大于minCommonWords）的关键帧进行比较
//         if(pKFi->mnLoopWords>minCommonWords)
//         {
//             nscores++;// 这个变量后面没有用到

//             // 用mBowVec来计算两者的相似度得分
//             float si = mpVoc->score(pKF->mBowVec,pKFi->mBowVec);

//             pKFi->mLoopScore = si;
//             if(si>=minScore)
//                 lScoreAndMatch.push_back(make_pair(si,pKFi));
//         }
//     }

//     // 如果没有超过指定相似度阈值的，那么也就直接跳过去
//     if(lScoreAndMatch.empty())
//         return vector<KeyFrame*>();


//     list<pair<float,KeyFrame*> > lAccScoreAndMatch;
//     float bestAccScore = minScore;

//     // Lets now accumulate score by covisibility
//     // 单单计算当前帧和某一关键帧的相似性是不够的，这里将与关键帧相连（权值最高，共视程度最高）的前十个关键帧归为一组，计算累计得分
//     // Step 4：计算上述候选帧对应的共视关键帧组的总得分，得到最高组得分bestAccScore，并以此决定阈值minScoreToRetain
//     for(list<pair<float,KeyFrame*> >::iterator it=lScoreAndMatch.begin(), itend=lScoreAndMatch.end(); it!=itend; it++)
//     {
//         KeyFrame* pKFi = it->second;
//         vector<KeyFrame*> vpNeighs = pKFi->GetBestCovisibilityKeyFrames(10);

//         float bestScore = it->first; // 该组最高分数
//         float accScore = it->first;  // 该组累计得分
//         KeyFrame* pBestKF = pKFi;    // 该组最高分数对应的关键帧
//         // 遍历共视关键帧，累计得分 
//         for(vector<KeyFrame*>::iterator vit=vpNeighs.begin(), vend=vpNeighs.end(); vit!=vend; vit++)
//         {
//             KeyFrame* pKF2 = *vit;
//             // 只有pKF2也在闭环候选帧中，且公共单词数超过最小要求，才能贡献分数
//             if(pKF2->mnLoopQuery==pKF->mnId && pKF2->mnLoopWords>minCommonWords)
//             {
//                 accScore+=pKF2->mLoopScore;
//                 // 统计得到组里分数最高的关键帧
//                 if(pKF2->mLoopScore>bestScore)
//                 {
//                     pBestKF=pKF2;
//                     bestScore = pKF2->mLoopScore;
//                 }
//             }
//         }

//         lAccScoreAndMatch.push_back(make_pair(accScore,pBestKF));
//         // 记录所有组中组得分最高的组，用于确定相对阈值
//         if(accScore>bestAccScore)
//             bestAccScore=accScore;
//     }

//     // Return all those keyframes with a score higher than 0.75*bestScore
//     // 所有组中最高得分的0.75倍，作为最低阈值
//     float minScoreToRetain = 0.75f*bestAccScore;

//     set<KeyFrame*> spAlreadyAddedKF;
//     vector<KeyFrame*> vpLoopCandidates;
//     vpLoopCandidates.reserve(lAccScoreAndMatch.size());

//     // Step 5：只取组得分大于阈值的组，得到组中分数最高的关键帧作为闭环候选关键帧
//     for(list<pair<float,KeyFrame*> >::iterator it=lAccScoreAndMatch.begin(), itend=lAccScoreAndMatch.end(); it!=itend; it++)
//     {
//         if(it->first>minScoreToRetain)
//         {
//             KeyFrame* pKFi = it->second;
//             // spAlreadyAddedKF 是为了防止重复添加
//             if(!spAlreadyAddedKF.count(pKFi))
//             {
//                 vpLoopCandidates.push_back(pKFi);
//                 spAlreadyAddedKF.insert(pKFi);
//             }
//         }
//     }

//     return vpLoopCandidates;
// }


// bool DetectLoop(){
    
//     // Compute reference BoW similarity score
//     // This is the lowest score to a connected keyframe in the covisibility graph
//     // We will impose loop candidates to have a higher similarity than this
//     const vector<Mat> vpConnectedKeyFrames;
//     const Mat mpCurrentKF;
//     const DBoW2::BowVector &CurrentBowVec = mpCurrentKF->mBowVec;
//     float minScore = 1;
//     for(size_t i=0; i<vpConnectedKeyFrames.size(); i++)
//     {
//         KeyFrame* pKF = vpConnectedKeyFrames[i];
//         if(pKF->isBad())
//             continue;
//         const DBoW2::BowVector &BowVec = pKF->mBowVec;

//         float score = mpORBVocabulary->score(CurrentBowVec, BowVec);

//         if(score<minScore)
//             minScore = score;
//     }

//     // Query the database imposing the minimum score
//     vector<KeyFrame*> vpCandidateKFs = mpKeyFrameDB->DetectLoopCandidates(mpCurrentKF, minScore);
// }


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
/////asdfasdf
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

int main(){

    // cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    // Vocabulary used for relocalization.
    ///用于重定位的ORB特征字典
    // ORBVocabulary* mpORBvocabulary;

    //建立一个新的ORB字典
    // mpVocabulary = new ORB_SLAM2::ORBVocabulary();

    // ORB_SLAM2::ORBVocabulary mpVocabulary;
    // ReadVocabulary(mpVocabulary);

    //读取图片
    cout<<"reading images..."<<endl;
    vector<cv::Mat> vmImages;
    // cv::String img_path = "/home/jialin/Documents/VSC_Projects/BoW_ORB/data/*.png";
    cv::String img_path = "/media/jialin/045E58135E57FC3C/UBUNTU/Long_term_VL_dataset/Oxford/*.jpg";
    ReadImg(vmImages, img_path);
    cout<<vmImages.size()<<endl;

    //提取ORB descripter
    cout<<"extracting ORB descriptors..."<<endl;
    vector<cv::Mat> vmDescriptors;
    ExtractORB(vmImages, vmDescriptors);
    // cout<<vmDescriptors[0].rowRange(0,5).colRange(0,5)<<endl;
    
    cout<<"building vocabulary..."<<endl;

    // string f_path = "/home/jialin/Documents/VSC_Projects/BoW_ORB/Vocabulary/vocabulary.yml.gz";
    // mpVocabulary->load(f_path);
    BuildVoc(mpVocabulary, vmDescriptors);

    cout<<"saving vocabulary..."<<endl;
    mpVocabulary -> save("vocabulary.yml.gz");

    cout<<*mpVocabulary<<endl;

    //
    // cout<<"computing BoW..."<<endl;
    
    // DBoW2::BowVector CBowVec = ComputeBoW(vmDescriptors[0]);
    // cout<<CBowVec.size()<<endl;
    // cout<<CBowVec[0]<<endl;
    // cout<<CBowVec[1]<<endl;
    // cout<<"comparing score ..."<<endl;
    // for(size_t i=0; i<vmDescriptors.size(); i++){
    //     DBoW2::BowVector mBowVec = ComputeBoW(vmDescriptors[i]);
    //     float score = mpVocabulary->score(CBowVec, mBowVec);
    //     cout<<i<<", "<<score<<endl;
    // }

    delete mpVocabulary;
    return 0;
}




