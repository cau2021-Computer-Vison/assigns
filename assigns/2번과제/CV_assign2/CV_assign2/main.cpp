/**
  @file videocapture_basic.cpp
  @brief A very basic sample for using VideoCapture and VideoWriter
  @author PkLab.net
  @date Aug 24, 2016
*/
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <stdio.h>

using namespace cv;
using namespace std;


Mat img1 = imread("img1.jpg", CV_16F);
Mat img2 = imread("img2.jpg", CV_16F);

vector<Point2f> src, dst;
vector<Point2f> newimg1_corners(4);


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

Mat toRectangle() {
    /* 이미지 1에 찍은 4개의 점으로 직각사각형을 만든다. */
    /* 이렇게 해야지 각 점에 대한 접근이 쉬워진다. */

    Size recSize(201, 201);
    Mat src_sq(recSize, img1.type());
    //Warping 후의 좌표
    // 직각사각형의 꼭지점이다. 이 4개의 점과 src의 4개의 점으로 행렬을 구한다.
    newimg1_corners[0] = Point2f(0, 0);
    newimg1_corners[1] = Point2f(src_sq.cols, 0);
    newimg1_corners[2] = Point2f(0, src_sq.rows);
    newimg1_corners[3] = Point2f(src_sq.cols, src_sq.rows);
    // 행렬 구하는 곳
    Mat trans = getPerspectiveTransform(src, newimg1_corners);
    // 행렬을 통해서 이미지 새로 채우기
    warpPerspective(img1, src_sq, trans, recSize);

    return src_sq;
}

Mat calHomography(vector<Point2f> ori, vector<Point2f> res) {
    Mat homo = findHomography(ori, res, 0);//최소자승법으로 homography 행렬 계산
    return homo;
}


float** calArrayHomography(vector<Point2f> ori, vector<Point2f> res) {
    // calHomography와 동일한 함수, float 
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

vector<int> calDotConversion(int x_Des, int y_Des, Mat Homo) { //특정 점에 대한 homography 변환 함수 출력(변환할 점의 x좌표, y좌표, homography 행렬)
    vector<int> Des;
    //Des.push_back(x_Des * Homo.at<double>(0, 0) + y_Des * Homo.at<double>(0, 1) + Homo.at<double>(0, 2));
    //Des.push_back(x_Des * Homo.at<double>(1, 0) + y_Des * Homo.at<double>(1, 1) + Homo.at<double>(1, 2));
    double x = x_Des * Homo.at<double>(0, 0) + y_Des * Homo.at<double>(0, 1) + Homo.at<double>(0, 2);
    double y = x_Des * Homo.at<double>(1, 0) + y_Des * Homo.at<double>(1, 1) + Homo.at<double>(1, 2);
    double c = x_Des * Homo.at<double>(2, 0) + y_Des * Homo.at<double>(2, 1) + Homo.at<double>(2, 2);
    Des.push_back((x / c));
    Des.push_back((y / c));
    return Des;
}


int main()
{
    cout << "img1" << img1.channels() << endl;
    cout << "img2" << img2.channels() << endl;

    while (src.size() < 4 || dst.size() < 4)
    {
        imshow("src", img1);
        imshow("dst", img2);
        setMouseCallback("src", Circles1);
        setMouseCallback("dst", Circles2);
        waitKey(100);
    }

    // src의 점 4개를 통해서 새로운 직각사각형의 그림을 만든다.
    Mat src_sq = toRectangle();

    Mat h = calHomography(newimg1_corners, dst);
    cout << h << endl;    //h 행렬(3x3) 한번 보기위해서

    if (src[1].y - src[0].y > dst[1].y - dst[0].y && src[2].x - src[0].x > dst[2].x - dst[0].x)
    {
        for (int i = 0; i < src_sq.rows; i++) {
            for (int j = 0; j < src_sq.cols; j++) {
                // 전방기하변환을 통한 행렬변환중
                vector<int> new_xy = calDotConversion(i, j, h);
                img2.at<Vec3b>(new_xy[1], new_xy[0])[0] = src_sq.at<Vec3b>(j, i)[0];
                img2.at<Vec3b>(new_xy[1], new_xy[0])[1] = src_sq.at<Vec3b>(j, i)[1];
                img2.at<Vec3b>(new_xy[1], new_xy[0])[2] = src_sq.at<Vec3b>(j, i)[2];
            }
        }
        imwrite("img_result.jpg", img2);        // img를 파일로 저장한다.
        imshow("result", img2);
        waitKey(0);
        return 0;
    }
    else {
        Mat empt(img2.size(), CV_8UC3);
        for (int i = 0; i < src_sq.rows; i++) {
            for (int j = 0; j < src_sq.cols; j++) {
                // 전방기하변환을 통한 행렬변환중
                vector<int> new_xy = calDotConversion(i, j, h);
                empt.at<Vec3b>(new_xy[1], new_xy[0])[0] = src_sq.at<Vec3b>(j, i)[0];
                empt.at<Vec3b>(new_xy[1], new_xy[0])[1] = src_sq.at<Vec3b>(j, i)[1];
                empt.at<Vec3b>(new_xy[1], new_xy[0])[2] = src_sq.at<Vec3b>(j, i)[2];
            }
        }
        dilate(empt, empt, Mat::ones(Size(3, 3), CV_8UC1), Point(-1, -1));

        for (int y = 0; y < empt.rows; y++) {
            for (int t = 0; t < empt.cols; t++) {
                //empt에서 비어 있지 않은 화소 img2로 복사
                if (empt.at<Vec3b>(t, y)[0] > 0 || empt.at<Vec3b>(t, y)[1] > 0 || empt.at<Vec3b>(t, y)[2] > 0) {
                    img2.at<Vec3b>(t, y)[0] = empt.at<Vec3b>(t, y)[0];
                    img2.at<Vec3b>(t, y)[1] = empt.at<Vec3b>(t, y)[1];
                    img2.at<Vec3b>(t, y)[2] = empt.at<Vec3b>(t, y)[2];
                }
            }
        }
        imwrite("img_result.jpg", img2);        // img를 파일로 저장한다.
        imshow("tem", empt);
        imshow("result", img2);
        waitKey(0);


        return 0;
    }
}