#include <iostream>
#include <fstream>
#include <opencv4/opencv2/core.hpp>

using namespace std;
using namespace cv;

int main(){
    // ifstream inFile;
    // string kitti360 = "/media/jialin/045E58135E57FC3C/UBUNTU/KITTI360/";
    // inFile.open(kitti360+"data_poses/2013_05_28_drive_0007_sync/poses.txt");
    // if(!inFile){
    //     cout<<"unable to open file: poses.txt"<<endl;
    //     exit(1);
    // }

    // // while(!inFile.eof()){
    // string inLine;
    // for(int index=0; index<5; index++){
    //     getline(inFile, inLine,'\n');
    //     // cout << inLine <<endl;
    //     // stringstream strs(inLine);
    // }

/////////////////////////////////////////////////////////////////////////////////////////////
    //这个就不用解释了
                            // C1    C2    C3    C4         C1      C2      C3      C4
    // CV_8U 8位无符号整数        0     8     16    24      uchar    Vec2b    Vec3b  Vec4b  （0…..255）
    // CV_8S 8 位符号整数         1     9     17    25      char                            （-128…..127）
    // CV_16U 16 位无符号整数     2     10    18    26      ushort                          （0……65535）
    // CV_16S 16 位符号整数       3     11    19    27      short    Vec2s    Vec3s  Vec4s  （-32768…..32767）
    // CV_32S 32 位符号整数       4     12    20    28      int      Vec2i    Vec3i  Vec4i  （-2147483648……2147483647）
    // CV_32F 32 位浮点数         5     13    21    29      float    Vec2f    Vec3f  Vec4f  （-FLT_MAX ………FLT_MAX，INF，NAN)
    // CV_64F 64 位浮点数         6     14    22    30      double   Vec2d    Vec3d  Vec4d  （-DBL_MAX ……….DBL_MAX，INF，NAN)

    //https://udayawijenayake.com/2021/06/07/opencv-data-types/

    Mat E1 = Mat::eye(4, 4, CV_64F);
    Mat E2 = Mat::ones(4, 4, CV_64F);
    Mat E3 = Mat::zeros(4, 4, CV_64F);

    //这个我不理解，真的不理解
    Mat m1(((5, 2), CV_32F, Scalar(1,255,0)));
    //cout: [1;255;0;0], scalar里最多有四个数字，类型不限，输出总会是4x1的矩阵，一个channel，如果输入少于4个就用0补足
    //至于定义的size根本没用，输出总会是4x1的矩阵

    
    Mat m2(5, 2, CV_32FC2, Scalar(1,255,0));
    //输出是5x2的矩阵，channel数和设定的一样，e.g. CV_32F 或者 CV_32FC1 就是1个channel，全是1
    //如果CV_32FC2，那就是两个channel，还是5x2的矩阵，每个element是 1,255。三个channel的就不用说了吧


    Mat m3 = (Mat_<double>(1,3)<<1,2,3,4);
    //输出是1x3的矩阵，channel为1， [1,2,3], 输入数量大于矩阵大小时，后面的忽略。如果输入数量不足，则会用0补足


    int a[2][4] = {1,2,3,4,5};
    //cout<<a<<endl;输出的是地址， *a也是地址
    //输入的数据可以比数组size小，但不可以比数组size大
    //因我我没发输出他，所以不知道输出什么样
    Mat m4(2,3,CV_32S,a);
    //输出就是2x3的矩阵,1 channel,当a比矩阵size大时，多出的部分被忽略，比矩阵size小时，剩余会用0补足


    //用create创建矩阵
    Mat m5;
    m5.create(4,4,CV_8UC2);


    Scalar s(2,2,2,2);
    //输出就是[2,2,2,2]


    //这种是用指针，但是在这里我懒得演示了
    // IplImage* img = cvLoadImage("1.jpg",1);
    // Mat test(img); 


    // Mat tt(3,5, CV_64FC3, Scalar(1,2,3));
    // cout<<t<<endl;
    // cout<<t.at<double>(2,0);
    // tt=tt.reshape(1,(-1,3));
    // cout<<tt<<endl;
    // tt.row(0)-=t.at<double>(0,0);
    // tt.row(1)-=t.at<double>(1,0);


    // cout<<tt<<endl;
    // cout<<tt.size<<endl;
    // cout<<tt.channels()<<endl;

    //关于mat的属性的定义，比如dims, channels,可以看这个：
    //https://www.cnblogs.com/justkong/p/7278579.html


    //矩阵切片
    double m[3][4] = 
	{ {1, 2, 3,4},
	  {5,6,7,8},
	  {9,10,11,12},
	};
    // cout <<"array m: "<< m <<endl;
    cv::Mat testMat(3,4,CV_64F,m);
    cout << testMat << endl;

    Mat mm[20][20];
    for (int i=0; i<20; i++){
        for (int j=0; j<20; j++){
            mm[i][j] = Mat::ones(i,j,CV_16SC1);
        }
    }

    cout<<mm[1][3]<<endl;

    

    // Mat mat = Mat::eye(Size(12,12), CV_8UC1);
    // FileStorage fs("vocabulary.xml", FileStorage::WRITE);
    // fs<<"vocabulary"<<mat;
    // fs.release();

    

    // Mat part = testMat.rowRange(0,2);
    // cout<<part<<endl;

    // part.copyTo(testMat.rowRange(1,3));
    // cout<<testMat<<endl;


    // Mat tt(4,5,CV_64FC3,Scalar(1,2,3));
    // Mat tt = Mat::zeros(4,5,CV_64FC3);
    // cout<<tt<<endl;
    // // cout<<tt.at<double>(3,3)<<endl;

    // // tt.at<Vec3d>(1,1) = testMat.rowRange(0,3).col(0);
    // testMat.rowRange(0,3).col(0).copyTo(tt.at<Vec3d>(1,1));

    // cout<<tt.at<Vec3d>(1,1)<<endl;

    // Mat chan[1];
    // split(tt, chan);
    // cout<<chan[2]<<endl;
    // cout<<chan[0].channels()<<endl;

    // vector<Mat> mer;
    // Mat merged;
    // mer.push_back(chan[0]);
    // mer.push_back(chan[2]);
    // merge(mer,merged);
    // cout<<merged<<endl;
    // cout<<merged.size<<endl;
    // cout<<merged.channels()<<endl;


    // string st = "asd";
    // cout<<st.length()<<endl;

    // bool larg = testMat.at<double>(2,2)>2000;
    // cout<<larg<<endl;

    // testMat.convertTo(testMat, CV_16S);
    // cout<<testMat<<endl;
    // cout<<testMat.type()<<endl;

    // int inn=200;
    // string ouu = to_string(inn);
    // // ouuinn;
    // cout<<"ouu: "<<ouu.length()<<endl;

    // string ooo = "20 -0.206532045 320.332 5666.112";
    // string::size_type idx;
    // idx = ooo.rfind("65",7);
    // if(idx!=string::npos){
    //     cout<<"idx: "<<idx<<endl;
    // }
    



    // Mat normalised(2,testMat.cols,testMat.type());
    // normalised.row(0) = testMat.row(0)/testMat.row(1);
    // normalised.row(1) = testMat.row(1)/testMat.row(0);

    // cout<<"normliased mat: "<< normalised<<endl;
    
    // normalised.convertTo(normalised, CV_16S);
    // cout<<"normlaised mat int: "<<normalised<<endl;

    // for(int i=0; i<normalised.rows; i++){
    //     for(int j=0; j<normalised.cols; j++){
    //         cout<<normalised.at<short>(i,j)<<" ";
    //     }
    //     cout<<endl;
    // }

    // Mat asdf;
    // for(int i=0; i<testMat.cols; i++){
    //     if(testMat.at<float>(0,i)>0 && testMat.at<float>(0,i)<2.5){
    //         asdf.push_back(testMat.col(i).t());
    //     }
    // }
    // asdf = asdf.t();
    // cout<<"asdf: "<<asdf<<endl;






    // // cv::Mat A=testMat(cv::Rect(0,0,2,3));
    // // cout << A << endl;

    // Mat mask = testMat.row(2)>10;
    // Mat mask_all;
    // mask_all.push_back(mask);
    // mask_all.push_back(mask);
    // mask_all.push_back(mask);
    // cout<<mask_all<<endl;
    // Mat f;
    // testMat.copyTo(f,mask_all);
    // cout<<f<<endl;
    // // cout<<f(mask_all)<<endl;
    // vector<int>idx;
    // sortIdx(f.row(2),idx, SORT_EVERY_ROW + SORT_ASCENDING);
    // for(int i=0; i<idx.size();i++){
    //     cout<<idx[i]<<endl;
    // }

    // sort()
    // findNonZero()
    // f.copyTo(sm);
    // cout<<sm<<endl;
    // // Mat mask2 = max(mask,200);
    // // cout<<mask2<<endl;
    // // Mat dst;
    // // testMat.copyTo(dst, mask);
    // // cout<<dst<<endl;
    // // Mat emp;
    // // testMat.convertTo(emp,CV_64F);
    // // cout <<emp<<endl;
    // testMat.convertTo(testMat, CV_64F);

    // Mat filter1;
    // for(int i=0; i<testMat.cols; i++){
    //     // cout<<testMat.at<double>(2,i)<<endl;
    //     // filter1.push_back(testMat.col(i).t());
    //     if(testMat.at<double>(2,i)>7){
            
    //         filter1.push_back(testMat.col(i).t());
    //     }
        
    // }
    // filter1 = filter1.t();
    // // cout<<filter1<<endl;
    // // cout<<filter1.size<<endl;
    // bool mtype = testMat.type()==6;
    // // cout<<mtype<<endl;


    




    return 0;
}
