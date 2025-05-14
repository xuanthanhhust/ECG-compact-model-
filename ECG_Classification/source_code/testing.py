import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from PIL import Image
import os
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import numpy as np
from tqdm import tqdm

from Class_Dataset import MyDataset
from Class_Model import MyModule

###########################    CONFIGURE   #############################

DATA_ROOT_PATH = Path("E:/pv/WORKING/ECG_main_folder/ECG_Classification/data")
MODEL_PATH = Path("E:/pv/WORKING/ECG_main_folder/ECG_Classification/model_save/ecg_model.pth")
BATCH_SIZE = 4

# ====================== Inference & Evaluation ======================

def evaluate():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load model
    model = MyModule()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    # Transform
    transform = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.ToTensor()
    ])

    # Load test data
    test_dataset = MyDataset(root_dir=DATA_ROOT_PATH/'test_images', transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    all_preds = []
    all_labels = []

    #Là chạy mô hình trên tập test, lấy dự đoán (preds) và so sánh với nhãn thật (labels)
    with torch.no_grad(): #Tắt chế độ tính gradient để tiết kiệm bộ nhớ và tăng tốc, ko cần học 
        for inputs, labels in tqdm(test_loader, desc="🔍 Testing"): #inputs: batch ảnh scalogram, labels: nhãn tương ứng (class thực tế)
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs) #Dữ liệu đi qua mô hình (Nếu mô hình có lớp softmax/logits ở cuối, outputs sẽ là [batch_size, num_classes])
            _, preds = torch.max(outputs, 1) #Lấy chỉ số class có giá trị lớn nhất ở mỗi hàng → chính là class mà mô hình dự đoán
                #preds: vector dự đoán class (vd: [0, 1, 2, ...])

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            #preds và labels được chuyển về CPU và numpy để: Dễ dùng với sklearn (dùng tính accuracy, confusion matrix, F1-score).

    # Evaluation 
    print("✅ Evaluation Result:") #In ra tiêu đề
    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, digits=4)) #digits=4: làm tròn đến 4 chữ số thập phân

    #Tính Macro F1-score cho từng lớp, rồi lấy trung bình cộng
    print(f"F1 Score (macro): {f1_score(all_labels, all_preds, average='macro'):.4f}") 

    #Weighted F1: Tính F1-score từng lớp, nhưng cân theo số lượng mẫu ở mỗi lớp
    #Phù hợp khi các lớp mất cân bằng (unbalanced)
    print(f"F1 Score (weighted): {f1_score(all_labels, all_preds, average='weighted'):.4f}")

if __name__ == '__main__':
    evaluate()
