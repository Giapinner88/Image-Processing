import cv2

# Đọc ảnh
img = cv2.imread("image/test_image.jpg")

# Kiểm tra ảnh có load được không
if img is None:
    print("❌ Không thể đọc ảnh. Kiểm tra lại đường dẫn.")
    exit()

# Hiển thị thông tin cơ bản
print(f"📏 Kích thước ảnh: {img.shape[1]}x{img.shape[0]} pixel")

# Chuyển sang ảnh xám
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Hiển thị ảnh gốc và ảnh xám
cv2.imshow("Original", img)
cv2.imshow("Gray", gray)

# Chờ phím bất kỳ để thoát
cv2.waitKey(0)
cv2.destroyAllWindows()
