#include <opencv2/opencv.hpp>
//#include <opencv2/nonfree/nonfree.hpp>
#include "opencv2/xfeatures2d.hpp"

using namespace std;
using namespace cv;

int main(){

    Mat img = imread("/home/arch/CLionProjects/MYSFM/data/kxm1.jpg");
    Mat grayImage;
    cvtColor(img, grayImage, COLOR_BGR2GRAY);

//    Ptr<SIFT> detector = SIFT::create();
     Ptr<xfeatures2d::SURF> detector = xfeatures2d::SURF::create(400);
    // Ptr<xfeatures2d::StarDetector> detector = xfeatures2d::StarDetector::create(20, 20);
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

    std::vector<KeyPoint>  keypoints,keypoints2;
    detector->detect(img, keypoints);
    detector->detect(grayImage, keypoints2);


    // draw and show
    Mat img_keypoints,img_keypoints2;
    drawKeypoints(img, keypoints, img_keypoints);
    drawKeypoints(grayImage, keypoints2, img_keypoints2);
    imshow("SIFT", img_keypoints);
    imshow("grayImage_SIFT", img_keypoints2);

    waitKey();



    return 0;
}