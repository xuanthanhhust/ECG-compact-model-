import os
from tqdm import tqdm #tqdm dùng để hiện progress bar khi xử lý nhiều ảnh
import concurrent.futures
from functools import partial
import shutil
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor


# Đường dẫn gốc chứa 4 folder class
SOURCE_DIR = "E:/pv/WORKING/ECG_main_folder/ECG_Classification/data/processed_images"  
TRAIN_DIR = "E:/pv/WORKING/ECG_main_folder/ECG_Classification/data/train_images"
TEST_DIR = "E:/pv/WORKING/ECG_main_folder/ECG_Classification/data/test_images"
SPLIT_RATIO = 0.8  # 80% train, 20% test

# Bước 1: Lấy toàn bộ danh sách ảnh và nhãn
all_images = [] # list để chứa đường dẫn các ảnh trong file gốc 
all_labels = [] # list để chứa các label  (0, 1, 2, 3)

for label in os.listdir(SOURCE_DIR):
    label_path = os.path.join(SOURCE_DIR, label)
    if os.path.isdir(label_path):  # Chỉ xử lý nếu là folder
        for filename in os.listdir(label_path): # Duyệt qua từng file ảnh
            image_path = os.path.join(label_path, filename)
            all_images.append(image_path) # Thêm ảnh vào danh sách
            all_labels.append(label)   # Gán nhãn tương ứng

# Bước 2: Stratified split chia theo tỉ lệ 80 20 
train_imgs, test_imgs, train_labels, test_labels = train_test_split(
    all_images, all_labels, test_size=1 - SPLIT_RATIO, stratify=all_labels, random_state=4
)

# Bước 3: Hàm copy ảnh sang thư mục đích
def copy_image(img_path, label, target_root):
    label_dir = os.path.join(target_root, label) # Tạo đường dẫn tới folder class
    os.makedirs(label_dir, exist_ok=True)  
    filename = os.path.basename(img_path) # Lấy tên file gốc
    save_path = os.path.join(label_dir, filename) # Đường dẫn lưu ảnh

    shutil.copy(img_path, save_path) #shutil.copy để sao chép ảnh từ gốc sang thư mục mới

# Bước 4: Hàm dùng song song để copy nhiều ảnh nhanh
def copy_images_parallel(images, labels, target_root, max_workers=8):
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(tqdm(executor.map(lambda args: copy_image(*args), zip(images, labels, [target_root]*len(images))), total=len(images)))

# Bước 5: Tiến hành copy song song
print("📂 Copy train images...")
copy_images_parallel(train_imgs, train_labels, TRAIN_DIR)

print("📂 Copy test images...")
copy_images_parallel(test_imgs, test_labels, TEST_DIR)

print(f"\n✅ Done! Train: {len(train_imgs)} ảnh | Test: {len(test_imgs)} ảnh")
