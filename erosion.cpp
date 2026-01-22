#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main() {
    // Đọc ảnh
    Mat img = imread("/home/giap-ros/Project/OpenCV/image/test_image.jpg", IMREAD_GRAYSCALE);
    if (img.empty()) {
        cerr << "❌ Không thể đọc ảnh." << endl;
        return -1;
    }

    // Tạo structuring element 3x3
    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));

    // Áp dụng erosion
    Mat eroded;
    erode(img, eroded, kernel, Point(-1, -1), 1);

    // Hiển thị ảnh
    imshow("Original (Gray)", img);
    imshow("Eroded", eroded);

    cout << "Nhấn ESC để thoát..." << endl;
    while (true) {
        if (waitKey(10) == 27) break;
    }

    destroyAllWindows();
    return 0;
}
