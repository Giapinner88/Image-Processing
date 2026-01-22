import matplotlib.pyplot as plt
import cv2
import glob

# Lấy danh sách ảnh đã vẽ lines
images = glob.glob('calibration_images/*_drawn.png')[:20] # Lấy tối đa 20 ảnh

plt.figure(figsize=(15, 10)) # Kích thước ảnh to

for i, image_file in enumerate(images):
    img = cv2.imread(image_file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Chuyển BGR sang RGB để hiển thị đúng màu
    
    plt.subplot(4, 5, i+1) # Tạo lưới 4 hàng, 5 cột
    plt.imshow(img)
    plt.title(f"Mẫu {i+1}")
    plt.axis('off') # Tắt trục toạ độ cho đẹp

plt.tight_layout()
plt.show()