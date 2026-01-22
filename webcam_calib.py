import numpy as np
import cv2
import time

# --- CẤU HÌNH ---
# Kích thước bàn cờ (số điểm giao nhau bên trong: hàng, cột)
# Ví dụ: Bàn cờ 10x7 ô vuông thì thường có 9x6 điểm giao nhau.
chessboardSize = (9, 6)
# Kích thước khung hình camera (sẽ tự cập nhật khi bật cam)
frameSize = (640, 480)

# Ngưỡng dừng việc tìm kiếm điểm (criteria)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Chuẩn bị các điểm toạ độ 3D trong thế giới thực (0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1, 2)

# Mảng lưu trữ điểm 3D và 2D từ tất cả các ảnh chụp được
objpoints = [] # Điểm 3D ngoài thực tế
imgpoints = [] # Điểm 2D trên mặt phẳng ảnh

# Mở Webcam (số 0 thường là webcam laptop)
cap = cv2.VideoCapture(0)

print("HƯỚNG DẪN:")
print("- Di chuyển bàn cờ trước camera.")
print("- Nhấn phím 's' để LƯU mẫu khi thấy các góc được vẽ màu.")
print("- Cần khoảng 15-20 mẫu ở các góc độ khác nhau.")
print("- Nhấn phím 'c' để BẮT ĐẦU HIỆU CHỈNH (Calibrate).")
print("- Nhấn 'q' để THOÁT.")

count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frameSize = (frame.shape[1], frame.shape[0])
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Tìm góc bàn cờ
    ret, corners = cv2.findChessboardCorners(gray, chessboardSize, None)

    # Nếu tìm thấy, vẽ lên màn hình để người dùng biết
    display_frame = frame.copy()
    if ret == True:
        cv2.drawChessboardCorners(display_frame, chessboardSize, corners, ret)
        cv2.putText(display_frame, "Da tim thay! Nhan 's' de luu.", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.putText(display_frame, f"So luong mau da luu: {count}", (10, 450), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow('Camera Calibration', display_frame)

    key = cv2.waitKey(1)

    # Nhấn 's' để lưu điểm dữ liệu
    if key == ord('s') and ret == True:
        objpoints.append(objp)
        # Tinh chỉnh toạ độ góc chính xác hơn (subpixel)
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)
        count += 1
        print(f"Đã lưu mẫu số {count}")
        time.sleep(0.5) # Dừng một chút để tránh bấm đúp

    # Nhấn 'c' để bắt đầu tính toán
    elif key == ord('c'):
        if count < 10:
            print("!!! Cần ít nhất 10 mẫu để hiệu chỉnh chính xác. Hãy chụp thêm.")
        else:
            print("--- Đang tính toán hiệu chỉnh... Vui lòng đợi ---")
            
            # Hàm quan trọng nhất: calibrateCamera
            ret, cameraMatrix, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, frameSize, None, None)

            print("\n=== KẾT QUẢ HIỆU CHỈNH ===")
            print("Reprojection Error (Sai số): ", ret)
            print("\nCamera Matrix (Ma trận nội tại):")
            print(cameraMatrix)
            print("\nDistortion Coefficients (Hệ số méo):")
            print(dist)
            
            # Lưu kết quả ra file để dùng lần sau
            np.savez("camera_calib_data.npz", cameraMatrix=cameraMatrix, dist=dist)
            print("\nĐã lưu thông số vào file 'camera_calib_data.npz'")
            
            print("\nChuyển sang chế độ xem thử (Undistortion Preview)...")
            break

    # Nhấn 'q' để thoát
    elif key == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        exit()

# --- PHẦN XEM THỬ KẾT QUẢ SAU KHI HIỆU CHỈNH ---
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    h, w = frame.shape[:2]
    # Lấy ma trận camera tối ưu mới (cắt bỏ vùng đen nếu muốn)
    newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, dist, (w,h), 1, (w,h))

    # Khử méo (Undistort)
    dst = cv2.undistort(frame, cameraMatrix, dist, None, newCameraMatrix)

    # Cắt ảnh theo vùng ROI (nếu cần thiết)
    # x, y, w, h = roi
    # dst = dst[y:y+h, x:x+w]

    # Hiển thị song song: Trái (Gốc) - Phải (Đã sửa)
    combined = np.hstack((frame, dst))
    cv2.imshow('Original (Left) vs Undistorted (Right) - Press q to quit', combined)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()