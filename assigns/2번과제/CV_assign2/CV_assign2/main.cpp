/**
  @file videocapture_basic.cpp
  @brief A very basic sample for using VideoCapture and VideoWriter
  @author PkLab.net
  @date Aug 24, 2016
*/
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
using namespace cv;
using namespace std;

Mat img1 = imread("img1.jpg");
Mat img2 = imread("img2.jpg");

vector<Point2f> src, dst;


void Circles1(int event, int x, int y, int flags, void* param) {

    switch (event)
    {
    case EVENT_FLAG_LBUTTON:
        circle(img1, Point(x, y), 10, Scalar(0, 0, 255), 2, 8);
        src.push_back(Point2f(x, y));
    }
}
void Circles2(int event, int x, int y, int flags, void* param) {

    switch (event)
    {
    case EVENT_FLAG_LBUTTON:
        circle(img2, Point(x, y), 10, Scalar(0, 0, 255), 2, 8);
        dst.push_back(Point2f(x, y));
    }
}

Mat calHomography(vector<Point2f> ori, vector<Point2f> res) {
    Mat homo = findHomography(ori, res, 0);//�ּ��ڽ¹����� homography ��� ���
    return homo;
}


float** calArrayHomography(vector<Point2f> ori, vector<Point2f> res) {
    // calHomography�� ������ �Լ�, float 
    Mat homo = findHomography(ori, res, 0);
    float** array2d = 0;
    array2d = new float* [3];

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            array2d[i][j] = homo.at<float>(i, j);
        }
    }
    return array2d;
}

vector<float> calDotConversion(float x_Des, float y_Des, Mat Homo) { //Ư�� ���� ���� homography ��ȯ �Լ� ���(��ȯ�� ���� x��ǥ, y��ǥ, homography ���)
    vector<float> Des;
    Des.push_back(x_Des * Homo.at<float>(0, 0) + y_Des * Homo.at<float>(0, 1) + Homo.at<float>(0, 2));
    Des.push_back(x_Des * Homo.at<float>(1, 0) + y_Des * Homo.at<float>(1, 1) + Homo.at<float>(1, 2));
    return Des;
}


int main()
{

    while (src.size() < 4 || dst.size() < 4)
    {
        imshow("src", img1);
        imshow("dst", img2);
        setMouseCallback("src", Circles1);
        setMouseCallback("dst", Circles2);
        waitKey(100);
    }
    return 0;
}