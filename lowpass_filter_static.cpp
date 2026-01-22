#include <opencv2/opencv.hpp>
#include <iostream>
using namespace cv;
using namespace std;

int main() {
    // Đọc ảnh
    Mat img = imread("/home/giap-ros/Project/OpenCV/image/test_image.jpg");
    if (img.empty()) {
        cerr << "❌ Không thể đọc ảnh." << endl;
        return -1;
    }

    Mat blurred_box, blurred_gaussian, blurred_bilateral;

    // 1️⃣ Box Filter (Average Blur) - dạng cơ bản nhất
    blur(img, blurred_box, Size(5, 5));

    // 2️⃣ Gaussian Blur - giảm nhiễu mượt hơn
    GaussianBlur(img, blurred_gaussian, Size(5, 5), 0);

    // 3️⃣ Bilateral Filter - giữ biên, giảm nhiễu tốt
    bilateralFilter(img, blurred_bilateral, 9, 75, 75);

    // Hiển thị ảnh
    imshow("Original", img);
    imshow("Box Filter", blurred_box);
    imshow("Gaussian Filter", blurred_gaussian);
    imshow("Bilateral Filter", blurred_bilateral);

    cout << "Nhấn ESC để thoát..." << endl;
    while (true) {
        if (waitKey(10) == 27) break;
    }

    destroyAllWindows();
    return 0;
}
