import torch
import os

print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("cuDNN version:", torch.backends.cudnn.version())


print("CUDA Available:", torch.cuda.is_available())
print("CUDA Device Count:", torch.cuda.device_count())
print("Current Device:", torch.cuda.current_device())
print("Device Name:", torch.cuda.get_device_name(0))

# 경로 확인
model_path = r'C:\python_project\CV_drowsy_detect\defalt_weight\yolov10n.pt'
data_path = r'C:\python_project\CV_drowsy_detect\drowsy detection.v2i.yolov8\data.yaml'

print("Model Path Exists:", os.path.exists(model_path))
print("Data Path Exists:", os.path.exists(data_path))
