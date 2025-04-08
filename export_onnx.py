import os
import torch
from PIL import Image
from efficientnet_pytorch import EfficientNet as enet
from torchvision import transforms as tf

INPUT_IMAGE_SIZE = 224

os.makedirs('models', exist_ok=True)

modelver = ('b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7')[0]
model = enet.from_pretrained(f'efficientnet-{modelver}')
model = model.cuda()

onnx_filename = f"models/efnet-{modelver}.onnx"
if not os.path.exists(onnx_filename):
    print("Building ONNX & TensorRT model\nThis will take about 10 min or more")
    dummy_input = torch.randn(1, 3, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE).cuda()
    model.set_swish(memory_efficient=False)
    torch.onnx.export(model, dummy_input, onnx_filename, verbose=True)
