#include <opencv2/opencv.hpp>
#include <iostream>
using namespace cv;
using namespace std;

int main() {
    // Đọc ảnh
    Mat img = imread("/home/giap-ros/Project/OpenCV/image/test_image.jpg");

    if (img.empty()) {
        cerr << "❌ Không thể đọc ảnh. Kiểm tra lại đường dẫn." << endl;
        return -1;
    }

    cout << "📏 Kích thước ảnh: " << img.cols << "x" << img.rows << " pixel" << endl;

    // Chuyển sang ảnh xám
    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);

    // Hiển thị ảnh
    imshow("Original", img);
    imshow("Gray", gray);

    waitKey(0);
    destroyAllWindows();
    return 0;
}
