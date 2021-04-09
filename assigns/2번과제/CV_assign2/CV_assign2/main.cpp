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


int main(){
    cout << "OpenCV Version : " << CV_VERSION << endl;

    Mat first, second;
    first = imread("src.jpg", IMREAD_GRAYSCALE);
    second = imread("dst.jpg", IMREAD_GRAYSCALE);
    if (first.empty() || second.empty()) {
        cout << "Image Load Error!" << endl;
    }
    resize(first, first, Size(), 0.25, 0.25);
    resize(second, second, Size(), 0.25, 0.25);


    return 0;
}
