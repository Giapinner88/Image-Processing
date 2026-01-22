import cv2
import numpy as np

# --- CẤU HÌNH ---
ARUCO_DICT = cv2.aruco.DICT_5X5_1000
MARKER_ID = 0          # ID của Marker (0, 1, 2...)
MARKER_SIZE = 500      # Kích thước phần mã (pixels)
BORDER_SIZE = 100      # Độ dày viền trắng (pixels) - Nên để dày một chút cho dễ nhìn

def create_marker_with_border():
    # 1. Load từ điển
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)

    # 2. Tạo ảnh Marker gốc (chỉ có đen trắng, sát viền)
    img_marker = cv2.aruco.generateImageMarker(aruco_dict, MARKER_ID, MARKER_SIZE)

    # 3. Thêm viền trắng (Padding) bằng hàm copyMakeBorder
    # Đối số: ảnh gốc, top, bottom, left, right, kiểu viền, màu sắc
    img_final = cv2.copyMakeBorder(
        img_marker, 
        BORDER_SIZE, BORDER_SIZE, BORDER_SIZE, BORDER_SIZE, 
        cv2.BORDER_CONSTANT, 
        value=[255, 255, 255] # Màu trắng (White)
    )

    # 4. Lưu ảnh
    filename = f"marker_id_{MARKER_ID}_iphone.png"
    cv2.imwrite(filename, img_final)
    print(f"Đã tạo: {filename}")
    print(f"Kích thước cuối cùng: {img_final.shape[1]}x{img_final.shape[0]} pixels")

    # Hiển thị kiểm tra
    cv2.imshow("Marker cho iPhone", img_final)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    create_marker_with_border()