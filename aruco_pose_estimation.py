import numpy as np
import cv2
import sys
import os
import time

# --- CẤU HÌNH ---
MARKER_SIZE = 0.05       # 0.05 mét = 5 cm (ĐO LẠI MARKER CỦA BẠN)
ARUCO_DICT_TYPE = cv2.aruco.DICT_5X5_1000
CALIB_FILE = "camera_calib_data.npz"

def load_camera_calibration(filename):
    if not os.path.exists(filename):
        print(f"LỖI: Không tìm thấy file '{filename}'! Hãy chạy calib trước.")
        return None, None
    data = np.load(filename)
    return data["cameraMatrix"], data["dist"]

def get_marker_points(size):
    # Định nghĩa 4 đỉnh 3D của marker (Z=0)
    half_size = size / 2.0
    return np.array([
        [-half_size, half_size, 0],
        [half_size, half_size, 0],
        [half_size, -half_size, 0],
        [-half_size, -half_size, 0]
    ], dtype=np.float32)

def main():
    # 1. Load thông số camera
    camera_matrix, dist_coeffs = load_camera_calibration(CALIB_FILE)
    if camera_matrix is None:
        sys.exit(1)

    # 2. Khởi tạo Camera & Detector
    cap = cv2.VideoCapture(0)
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_TYPE)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    # Điểm 3D chuẩn dùng cho solvePnP
    obj_points = get_marker_points(MARKER_SIZE)

    print(f"--- ĐANG CHẠY ---")
    print(f"Marker Size cấu hình: {MARKER_SIZE}m")
    print("Nhấn 's': CHỤP ẢNH để làm báo cáo.")
    print("Nhấn 'q': THOÁT.")

    screenshot_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = detector.detectMarkers(gray)

        if ids is not None:
            # Vẽ viền marker
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            for i in range(len(ids)):
                # Tìm Pose (Vị trí & Góc xoay)
                marker_corners_2d = corners[i].reshape((4, 2))
                success, rvec, tvec = cv2.solvePnP(
                    obj_points, 
                    marker_corners_2d, 
                    camera_matrix, 
                    dist_coeffs
                )

                if success:
                    # Vẽ trục 3D (X:Đỏ, Y:Lục, Z:Lam)
                    cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.03)

                    # Tính khoảng cách
                    x, y, z = tvec[0][0], tvec[1][0], tvec[2][0]
                    distance = np.sqrt(x**2 + y**2 + z**2)

                    # Hiển thị text lên màn hình
                    text_pos = (int(marker_corners_2d[0][0]), int(marker_corners_2d[0][1]) - 10)
                    cv2.putText(frame, f"ID:{ids[i][0]} Dist:{distance:.2f}m", 
                                text_pos, cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 255, 0), 2)

        # Hiển thị frame
        cv2.imshow('ArUco Pose Estimation (Press s to Save)', frame)

        # --- XỬ LÝ PHÍM BẤM ---
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'): # Thoát
            break
        elif key == ord('s'): # Chụp ảnh
            screenshot_count += 1
            # Đặt tên file theo thời gian để không bị trùng
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"aruco_report_{timestamp}.png"
            
            cv2.imwrite(filename, frame)
            print(f"--> Đã lưu ảnh: {filename}")
            
            # Nháy màn hình một chút để biết là đã chụp (hiệu ứng visual)
            cv2.imshow('ArUco Pose Estimation (Press s to Save)', 255 - frame)
            cv2.waitKey(50)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()