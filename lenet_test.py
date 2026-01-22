import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np

# --- 1. CẤU HÌNH ---
BATCH_SIZE = 64
EPOCHS = 2  # Chạy thử 2 vòng cho nhanh
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. CHUẨN BỊ DỮ LIỆU ---
# Resize về 32x32 cho đúng chuẩn LeNet gốc
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='./data', train=False, transform=transform)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

# --- 3. MÔ HÌNH LENET-5 ---
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5) # Flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = LeNet5().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# --- 4. HUẤN LUYỆN (TRAINING) ---
print(f"Đang chạy trên: {DEVICE}")
print("Bắt đầu train... (Vui lòng đợi 1-2 phút)")

for epoch in range(EPOCHS):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 300 == 0:
            print(f'Epoch {epoch+1}/{EPOCHS} | Batch {batch_idx} | Loss: {loss.item():.4f}')

# --- 5. ĐÁNH GIÁ & HIỂN THỊ KẾT QUẢ ---
print("\nĐang kiểm tra độ chính xác...")
model.eval()
correct = 0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(DEVICE), target.to(DEVICE)
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True) # Lấy chỉ số có xác suất lớn nhất
        correct += pred.eq(target.view_as(pred)).sum().item()

acc = 100. * correct / len(test_loader.dataset)
print(f'\n=> Độ chính xác trên tập Test: {acc:.2f}%')

# --- 6. DEMO TRỰC QUAN (Visualizing Results) ---
print("Đang hiển thị một số ảnh dự đoán mẫu...")

# Lấy 1 batch ảnh từ tập test
images, labels = next(iter(test_loader))
images, labels = images.to(DEVICE), labels.to(DEVICE)

# Dự đoán
outputs = model(images)
_, preds = torch.max(outputs, 1)

# Vẽ 6 ảnh đầu tiên
fig = plt.figure(figsize=(10, 6))
for i in range(6):
    ax = fig.add_subplot(2, 3, i+1)
    # Chuyển về CPU và bỏ chuẩn hóa để hiển thị ảnh gốc cho đẹp
    img = images[i].cpu().squeeze().numpy()
    
    ax.imshow(img, cmap='gray')
    ax.set_title(f"Thực tế: {labels[i].item()} | Máy đoán: {preds[i].item()}", 
                 color=("green" if preds[i]==labels[i] else "red"))
    ax.axis('off')

plt.tight_layout()
plt.show()

print("\n------------------------------------------------")
print("BẮT ĐẦU PHÂN TÍCH ẢNH CỦA BẠN (CUSTOM IMAGE)")

# --- CẤU HÌNH INPUT ---
IMAGE_PATH = 'image/digit_number_2.png'  # Đảm bảo file này đang nằm cạnh file code
from PIL import Image, ImageOps # Import thêm thư viện xử lý ảnh

# --- HÀM XỬ LÝ ẢNH & DỰ ĐOÁN ---
def analyze_custom_image():
    # 1. Kiểm tra xem có file ảnh không
    import os
    if not os.path.exists(IMAGE_PATH):
        print(f"LỖI: Không tìm thấy file '{IMAGE_PATH}'.")
        print("Hãy mở Paint, vẽ một con số màu đen trên nền trắng, lưu lại tên là my_digit.png rồi chạy lại.")
        return

    # 2. Xử lý ảnh
    img = Image.open(IMAGE_PATH).convert('L') 
    img = ImageOps.invert(img)                
    img = img.resize((32, 32), Image.Resampling.BILINEAR) 
    
    # Chuyển sang Tensor
    img_tensor = transform(img).unsqueeze(0).to(DEVICE) 

    # 3. Dự đoán
    model.eval() 
    with torch.no_grad():
        logits = model(img_tensor)
        probs = F.softmax(logits, dim=1)
    
    # 4. Lấy dữ liệu
    top_p, top_class = probs.topk(1, dim=1)
    probabilities = probs.cpu().numpy().squeeze()
    prediction = top_class.item()
    confidence = top_p.item() * 100

    # 5. Lấy Feature Maps (Mổ xẻ bên trong)
    # Lưu ý: Các bước này nằm ngoài torch.no_grad() nên vẫn dính gradient
    act1 = F.relu(model.conv1(img_tensor)) 
    x = model.pool(act1)
    act2 = F.relu(model.conv2(x))

    # --- VẼ PLOT ---
    plt.figure(figsize=(12, 6))
    
    # Ảnh gốc
    plt.subplot(2, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title(f"Input\nDự đoán: {prediction} ({confidence:.1f}%)")
    plt.axis('off')

    # Biểu đồ xác suất
    plt.subplot(2, 3, 2)
    bars = plt.bar(range(10), probabilities, color='gray')
    bars[prediction].set_color('green') 
    plt.xticks(range(10))
    plt.title("Xác suất dự đoán")
    
    # Feature Map Conv1 (6 maps)
    plt.subplot(2, 3, 4)
    # FIX LỖI Ở ĐÂY: Thêm .detach() trước .cpu()
    act1_grid = np.concatenate([act1[0][i].detach().cpu().numpy() for i in range(6)], axis=1)
    plt.imshow(act1_grid, cmap='viridis')
    plt.title("Feature Maps: Conv1")
    plt.axis('off')

    # Feature Map Conv2 (16 maps)
    plt.subplot(2, 3, 5)
    # FIX LỖI Ở ĐÂY: Thêm .detach() trước .cpu()
    row1 = np.concatenate([act2[0][i].detach().cpu().numpy() for i in range(8)], axis=1)
    row2 = np.concatenate([act2[0][i].detach().cpu().numpy() for i in range(8, 16)], axis=1)
    act2_grid = np.concatenate((row1, row2), axis=0)
    plt.imshow(act2_grid, cmap='inferno')
    plt.title("Feature Maps: Conv2")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# Gọi hàm chạy
analyze_custom_image()