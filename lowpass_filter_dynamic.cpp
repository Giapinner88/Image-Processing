#include <opencv2/opencv.hpp>
#include <iostream>
using namespace cv;
using namespace std;

int main()
{
    VideoCapture cam(0, cv::CAP_V4L2);
    if (!cam.isOpened()) {
        cerr << "❌ Không mở được webcam." << endl;
        return -1;
    }

    Mat frame, filtered;
    while (true) {
        cam >> frame;
        if (frame.empty()) break;

        // Gaussian Low-pass
        GaussianBlur(frame, filtered, Size(7, 7), 0);

        imshow("Original", frame);
        imshow("Low-pass Filtered", filtered);

        if (waitKey(10) == 27) break; // ESC để thoát
    }

    cam.release();
    destroyAllWindows();
    return 0;
}