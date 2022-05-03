#include <opencv2/opencv.hpp>
//#include <opencv2/nonfree/nonfree.hpp>
#include "opencv2/xfeatures2d.hpp"

using namespace std;
using namespace cv;

int main(){

    Mat img1 = imread("/home/arch/CLionProjects/MYSFM/data/kxm1.jpg");
    Mat img2 = imread("/home/arch/CLionProjects/MYSFM/data/kxm2.jpg");
//    Mat grayImage;
//    cvtColor(img, grayImage, COLOR_BGR2GRAY);

//    Ptr<SIFT> detector = SIFT::create();
     Ptr<xfeatures2d::SURF> detector = xfeatures2d::SURF::create(400);
//     Ptr<xfeatures2d::StarDetector> detector = xfeatures2d::StarDetector::create(20, 20);
    // Ptr<ORB> detector = ORB::create(2000);
    // Ptr<BRISK> detector = BRISK::create();
    // Ptr<KAZE> detector = KAZE::create();
    // Ptr<AKAZE> detector = AKAZE::create();


//    //检测surf关键点、提取训练图像描述符
//    vector<KeyPoint> keyPoint;
//    Mat descriptor;
//    xfeatures2d::SurfFeatureDetector featureDetector(80);
//    featureDetector.detect(grayImage,keyPoint);
//    SurfDescriptorExtractor featureExtractor;
//    featureExtractor.compute(grayImage, keyPoint, descriptor);

    std::vector<KeyPoint>  keypoints1, keypoints2;
    Mat descriptorMat1, descriptorMat2;
    detector->detectAndCompute(img1, Mat(), keypoints1, descriptorMat1);
    detector->detectAndCompute(img2, Mat(), keypoints2, descriptorMat2);

    // draw and show
    Mat img_keypoints1,img_keypoints2;
    drawKeypoints(img1, keypoints1, img_keypoints1);
    drawKeypoints(img2, keypoints2, img_keypoints2);
    imshow("SIFT1", img_keypoints1);
    imshow("SIFT2", img_keypoints2);

    waitKey();

    //获得匹配特征点，并提取最优配对
//    cv::BFMatcher matcher;
    FlannBasedMatcher matcher;
    vector<DMatch> matchePoints;

    matcher.match(descriptorMat1, descriptorMat2, matchePoints, Mat());
    cout << "total match points: " << matchePoints.size() << endl;

    sort(matchePoints.begin(),matchePoints.end()); //按距离从小到大排序
    //提取强特征点
    //获取排在前N个的最优匹配结果
    vector<DMatch> goodMatchePoints;
    for(int i=0;i<50;i++)
    {
        goodMatchePoints.push_back(matchePoints[i]);
    }


    Mat img_match;
//    drawMatches(img1, keypoints1, img2, keypoints2, matchePoints, img_match);
//    drawMatches(img1, keypoints1, img2, keypoints2, goodMatchePoints, img_match, DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    drawMatches(img1,keypoints1,img2,keypoints2,goodMatchePoints,img_match,Scalar::all(-1),
                Scalar::all(-1),vector<char>(),DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    namedWindow("match", 0);
    imshow("match",img_match);
    imwrite("/home/arch/CLionProjects/MYSFM/output/match.jpg", img_match);

    waitKey();

    return 0;
}