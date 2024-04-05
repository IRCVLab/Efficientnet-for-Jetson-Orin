# EfficientNet Demo for Jetson Orin Nano

## RUN
python3 classification.py

## Inputs
 - Model Version: Enter Model Version of Efficientnet (b0~b7)
 - Model Type: Enter Model Type (Original, RT-FP16, RT-FP32
 - Input Image: Select Input Image (ex. wolf.jpg)

## Requirements(version will be added)
pytorch 1.13.0  
torchvision 0.14.1  
cv2 4.5.4  
pycuda 2024.1  
Efficientnet_pytorch 0.7.1  
tensorrt 8.5.2  
onnx 1.16.0  
