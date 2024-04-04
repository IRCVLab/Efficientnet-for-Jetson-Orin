import torch
from torchvision import transforms as tf
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tensorrt as trt
from efficientnet_pytorch import EfficientNet as enet
from onnx_trt import ONNXClassifierWrapper, convert_trt
import os
import time

def build_model(modelver, modeltp, cls_num):
    model = enet.from_pretrained(f'efficientnet-{modelver}')
    if modeltp == "Original":
        model = model.cuda()
    
    elif modeltp == "RT-FP16":
        model = build_onnx_trt(model, half=True, n_cls=cls_num)
    
    elif modeltp == "RT-FP32":
        model = build_onnx_trt(model, half=False, n_cls=cls_num)
    
    else:
        AssertionError("Model Type Error Please Enter Correct Type: Original, RT-FP16, RT-FP32")

    return model 

def build_onnx_trt(model, half, n_cls):

    model_path = os.path.abspath('models')
    file_list = os.listdir(model_path)
    onnx_filename = f"{model_path}/efnet-b1.onnx"

    # Build Pytorch Model -> Onnx Model
    if not any (files.endswith(".onnx") for files in file_list):
        dummy_input = torch.randn(1, 3, 224, 224)
        model.set_swish(memory_efficient=False)
        torch.onnx.export(model, dummy_input, onnx_filename, verbose=True)
    
    # Build Onnx Model -> TRT Model
    if half == True:
        trt_filename = f'{model_path}/trt_fp16.engine'
        if not any (files.endswith("16.engine") for files in file_list):
            convert_trt(onnx_filename, trt_filename, half)
    else :
        trt_filename = f'{model_path}/trt_fp32.engine'
        if not any (files.endswith("32.engine") for files in file_list):
            convert_trt(onnx_filename, trt_filename, half)

    batch_size = 1
    model = ONNXClassifierWrapper(trt_filename,[batch_size, n_cls],target_dtype=np.float32)

    return model

def image_preprocess(path, imgname):
    image_size=224
    img_trans=tf.Compose([tf.Resize(image_size), tf.CenterCrop(image_size),
                 tf.ToTensor(), tf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    img = Image.open(f"{path}/{imgname}")
    input_img = img_trans(img).unsqueeze(0)
    
    return input_img

def eval(model, modeltp, input_img):
    if modeltp == 'Original':
        model.eval()
        with torch.no_grad():
            startt = time.time()
            results = model(input_img.cuda())
            endt = time.time()
    else:
        startt = time.time()
        results=model.predict(input_img.numpy().transpose((0,3,2,1)))
        endt = time.time()
        results=torch.from_numpy(results)

    preds = torch.topk(results, k=3).indices.squeeze(0).tolist()
    inft = endt-startt
    print(f"\nInference Time: {inft*1000:.2f}ms\n")

    for idx in preds:
        label = labels_map[idx]
        prob = torch.softmax(results, dim=1)[0, idx].item()
        print(f'{label:<75} ({(prob*100):.2f}%)')

if __name__ == "__main__":
    current_dir = os.path.abspath('')

    print("EfficientNet Classification Demo\nThis model will claasify your input image among 1000 classes")
    modelver = input("\nEnter Model Version to use (b0~b7): ")
    modeltp = input("\nEnter Model type (Originial, RT-FP16, RT-FP32): ")
    image = input("\nEnter Image Name to Classify: ")

    labels_map=json.load(open(f'{current_dir}/labels_map.txt'))
    labels_map=[labels_map[str(i)] for i in range(1000)]

    model = build_model(modelver, modeltp, cls_num=len(labels_map))
    img = image_preprocess(current_dir, image)
    eval(model, modeltp, img)
