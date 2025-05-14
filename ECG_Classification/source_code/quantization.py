# quantization of the model

import torch
import torch.nn as nn
import torch.quantization
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from pathlib import Path

DATA_ROOT_PATH = Path("E:/pv/WORKING/ECG_main_folder/ECG_Classification/data")
QUANTIZERTED_MODEL_PATH = Path("E:/pv/WORKING/ECG_main_folder/ECG_Classification/model_save/ecg_model_quantizered.pth")
PRUNED_MODEL_PATH = Path("E:/pv/WORKING/ECG_main_folder/ECG_Classification/model_save/ecg_model_pruned.pth")

# Tải mô hình đã huấn luyện
model = torch.load(PRUNED_MODEL_PATH)

class QAT