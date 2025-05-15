# pruning.py

import torch
import torch.nn as nn
from Class_Model import MyModule
from Class_Dataset import MyDataset
from pathlib import Path
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from torch.backends import cudnn
import torchprune
from torchinfo import summary  # Thư viện để đo FLOPs và tham số

DATA_ROOT_PATH = Path("E:/pv/WORKING/ECG_main_folder/ECG_Classification/data")
MODEL_PATH = Path("E:/pv/WORKING/ECG_main_folder/ECG_Classification/model_save/ecg_model.pth")
COMPACT_MODEL_PATH = Path("E:/pv/WORKING/ECG_main_folder/ECG_Classification/model_save/ecg_model_compact.pth")
BATCH_SIZE = 4
def compute_fisher_information(model, data_loader, num_samples=100):
    """
    Tính toán Fisher Information cho các lớp Conv2D trong mô hình.
    
    model: Mô hình học sâu.
    data_loader: Dataloader của dữ liệu huấn luyện.
    num_samples: Số lượng mẫu dùng để tính Fisher.
    
    Trả về dictionary chứa Fisher information cho từng lớp Conv2D.
    """
    

    model.eval()
    fisher = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Khởi tạo Fisher information cho mỗi lớp Conv2D
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            fisher[name] = torch.zeros_like(module.weight.data)

    # Tính Fisher information
    samples = 0
    criterion = torch.nn.CrossEntropyLoss()
    for inputs, labels in data_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        model.zero_grad() # Đặt lại gradient về 0
        outputs = model(inputs) # Forward pass
        loss = criterion(outputs, labels) # Tính loss
        loss.backward() # Backward pass (tính gradient)

        # Cập nhật Fisher information cho các lớp Conv2D
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d) and module.weight.grad is not None:
                fisher[name] += module.weight.grad.detach() ** 2

        samples += 1
        if samples >= num_samples:
            break

    # Trung bình Fisher information qua các mẫu
    for name in fisher:
        fisher[name] /= samples

    return fisher

def fisher_prune_and_mask(model, fisher_info, prune_ratio=0.3):
    """
    Prune (cắt bớt) các bộ lọc trong các lớp Conv2D theo Fisher Information.
    
    model: Mô hình học sâu.
    fisher_info: Fisher information từ các lớp Conv2D.
    prune_ratio: Tỷ lệ cắt bớt các bộ lọc.
    
    Trả về một dictionary chứa mask (0/1) để loại bỏ bộ lọc.
    """
    masks = {}

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            weight = module.weight.data
            fisher = fisher_info[name]

            # Tính toán sự thay đổi mất mát cho mỗi bộ lọc
            delta_loss = 0.5 * (weight ** 2) * fisher
            delta_loss = delta_loss.view(weight.shape[0], -1).sum(dim=1)

            # Tìm các bộ lọc có delta_loss nhỏ nhất để prune
            num_filters_to_prune = int(prune_ratio * delta_loss.numel())
            _, prune_idx = torch.topk(delta_loss, num_filters_to_prune, largest=False)

            # Tạo mask cho các bộ lọc (bộ lọc nào được cắt bỏ thì set 0)
            mask = torch.ones(weight.shape[0], dtype=torch.bool)
            mask[prune_idx] = False
            masks[name] = mask

    return masks

def apply_pruning_mask(model, masks):
    """
    Áp dụng mask vào mô hình để loại bỏ các bộ lọc theo mask đã tính toán.
    
    model: Mô hình học sâu.
    masks: Mask cho các lớp Conv2D.
    
    Trả về mô hình sau khi pruning.
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            mask = masks[name]
            weight = module.weight.data
            module.weight.data = weight[mask]

    return model

def print_model_stats(model, input_size, note=""):
    info = summary(model, input_size=input_size, verbose=0)
    print(f"\n{note}Model stats:")
    print(f"  Parameters: {info.total_params:,}")
    print(f"  FLOPs: {info.total_mult_adds:,}")
    # Tính kích thước file model (MB)
    torch.save(model.state_dict(), "temp.pth")
    import os
    size_mb = os.path.getsize("temp.pth") / (1024 * 1024)
    os.remove("temp.pth")
    print(f"  Model size: {size_mb:.2f} MB")

def main():
    # Xác định thiết bị: nếu có GPU thì dùng, không thì dùng CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Khởi tạo mô hình và đưa lên device
    model = MyModule().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # In thông tin mô hình trước khi pruning   # In thông số mô hình trước pruning
    print_model_stats(model, input_size=(1, 1, 100, 100), note="Before pruning")
    # Tạo DataLoader cho dữ liệu huấn luyện
    transform = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.ToTensor()
    ])
    train_dataset = MyDataset(root_dir=DATA_ROOT_PATH/'train_images', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)


    # Fisher pruning với torch-pruning, chỉ áp dụng cho Conv2d
    example_inputs = torch.randn(1, 1, 100, 100).to(device)
    imp = tp.importance.FisherImportance()
    ignored_layers = [m for m in model.modules() if not isinstance(m, nn.Conv2d)]
    pruner = tp.pruner.MetaPruner(
        model,
        example_inputs,
        importance=imp,
        iterative_steps=1,
        pruning_ratio=0.3,  # Prune 30%
        ignored_layers=ignored_layers
    )
    pruner.step()  # Tự động prune và tái cấu trúc model

    print_model_stats(model, input_size=(1, 1, 100, 100), note="After pruning")

    torch.save(model.state_dict(), COMPACT_MODEL_PATH)
    print(f"\nPruned model saved to: {COMPACT_MODEL_PATH}")

if __name__ == "__main__":
    main()
