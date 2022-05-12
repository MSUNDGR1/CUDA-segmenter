#include <iostream>
#include <stdint.h>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "segmenter.cuh"

int xClick, yClick;
bool Clicked;

using namespace cv;
using namespace std;

void CallBackFunc(int event, int x, int y, int flags, void* userdata) {
    if (event == EVENT_LBUTTONDOWN) {
        xClick = x; yClick = y; Clicked = true;
        cout << "X: " << x << " Y: " << y << endl;
    }
}

int main(){
	Mat frame;
    VideoCapture vid0(0);
   

    vid0.set(CAP_PROP_FRAME_WIDTH, 640);
    vid0.set(CAP_PROP_FRAME_HEIGHT, 480);
   

    

    namedWindow("Cam 0", WINDOW_AUTOSIZE);

    namedWindow("Cam 1", WINDOW_AUTOSIZE);
    
    setMouseCallback("Cam 0", CallBackFunc, NULL);

    Clicked = false;
    while (true) {
        vid0 >> frame;
        //frame = imread("env_ex.png");


        if (!frame.empty()) {
            
            Mat gFrame;
            cvtColor(frame, gFrame, COLOR_BGR2GRAY);
            pair<Mat, int*> out = seg::segment(gFrame, 0.005, 40);
            frame = out.first;
            imshow("Cam 0", frame);
            imshow("Cam 1", gFrame);
            //pair<float,float> varmean = seg::variance(gFrame);
            //cout << "Variance: "<<varmean.first<<"  mean: "<<varmean.second<<endl;
            if (Clicked) {
                int realIndex = yClick * frame.cols + xClick;
                frame = seg::clusterIsolate(frame, out.second, 0, 255, 0, out.second[realIndex]);
                imshow("Cam 0", frame);
                Clicked = false;
                waitKey(0);
            }
            delete[] out.second;
        }

        int key = waitKey(30);
        if (key == ' ') break;
    }
    vid0.release();
    //vid1.release();
    destroyAllWindows();
  /* Mat testIm = imread("rovin_test.jpg");
    Mat gtest;
    cvtColor(testIm, gtest, COLOR_BGR2GRAY);
    namedWindow("Modded", WINDOW_AUTOSIZE);
    namedWindow("Unmodded", WINDOW_AUTOSIZE);
    //seg::meanTest(Mat());
    imshow("Unmodded", gtest);
    gtest = seg::segment(gtest, 0.02, 60);
    
    imshow("Modded", gtest);
    waitKey();*/

}