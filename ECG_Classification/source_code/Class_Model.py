import torch.nn.functional as F
from torch.backends import cudnn
import torch.nn as nn
import torch
import torchvision.models as models

#Cấu hình PyTorch
cudnn.benchmark = False
cudnn.deterministic = True #Đảm bảo rằng mô hình luôn tạo ra kết quả giống nhau khi huấn luyện lại.

torch.manual_seed(0)

# Xây dựng mô hình ResNet
class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        # Sử dụng ResNet18 làm backbone
        self.resnet = models.resnet18(pretrained=True)
        # Chỉnh sửa lớp đầu vào để nhận ảnh grayscale (1 kênh)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Chỉnh sửa lớp fully connected cuối cùng để ánh xạ vào 4 lớp output
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 4)
    
    def forward(self, x):
        x = self.resnet(x)
        return x


