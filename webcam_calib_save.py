import numpy as np
import cv2
import time
import os

# --- CẤU HÌNH ---
chessboardSize = (9, 6) # Sửa lại theo bàn cờ của bạn
frameSize = (640, 480)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# --- TẠO THƯ MỤC LƯU ẢNH ---
save_folder = 'calibration_images'
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
    print(f"Đã tạo thư mục '{save_folder}' để lưu ảnh.")

# Chuẩn bị toạ độ 3D
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1, 2)

objpoints = [] 
imgpoints = [] 

cap = cv2.VideoCapture(0)

print("--- HƯỚNG DẪN ---")
print("1. Đưa bàn cờ vào khung hình.")
print("2. Nhấn 's' để LƯU ẢNH (Code sẽ lưu cả ảnh gốc và ảnh có vẽ nét).")
print("3. Nhấn 'c' để bắt đầu tính toán hiệu chỉnh.")
print("4. Nhấn 'q' để thoát.")

count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frameSize = (frame.shape[1], frame.shape[0])
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Copy frame để vẽ lên mà không làm hỏng ảnh gốc
    display_frame = frame.copy()

    # Tìm góc bàn cờ
    ret, corners = cv2.findChessboardCorners(gray, chessboardSize, None)

    if ret == True:
        # Vẽ các góc tìm được lên display_frame
        cv2.drawChessboardCorners(display_frame, chessboardSize, corners, ret)
        
        # Hướng dẫn trên màn hình
        cv2.putText(display_frame, "Da tim thay! Nhan 's' de luu.", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.putText(display_frame, f"So luong mau da luu: {count}", (10, 450), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow('Camera Calibration', display_frame)

    key = cv2.waitKey(1)

    # --- LƯU ẢNH KHI NHẤN 'S' ---
    if key == ord('s') and ret == True:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)
        count += 1
        
        # 1. Lưu ảnh GỐC (để làm dữ liệu chuẩn)
        filename_raw = f"{save_folder}/img_{count}_raw.png"
        cv2.imwrite(filename_raw, frame)
        
        # 2. Lưu ảnh ĐÃ VẼ (để làm báo cáo cho đẹp)
        filename_drawn = f"{save_folder}/img_{count}_drawn.png"
        cv2.imwrite(filename_drawn, display_frame)
        
        print(f"--> Đã lưu: {filename_raw} và {filename_drawn}")
        
        # Hiệu ứng nháy màn hình báo hiệu đã chụp
        cv2.imshow('Camera Calibration', np.zeros_like(frame))
        cv2.waitKey(50) 
        time.sleep(0.5)

    elif key == ord('c'):
        if count < 10:
            print("!!! Cần ít nhất 10 mẫu. Hãy chụp thêm.")
        else:
            print("--- Đang tính toán... ---")
            ret_val, cameraMatrix, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, frameSize, None, None)
            
            print(f"Sai số (Reprojection Error): {ret_val}")
            np.savez("camera_calib_data.npz", cameraMatrix=cameraMatrix, dist=dist)
            print("Đã lưu thông số vào file.")
            break

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()