import torch
from torchvision import transforms as tf
import json
import numpy as np
import cv2
from PIL import Image
from efficientnet_pytorch import EfficientNet as enet
from onnx_trt import ONNXClassifierWrapper, convert_trt
import os
import time
#import torch.autograd.profiler as profiler
#import nvtx

#@nvtx.annotate("building model", color='blue')
def build_model(modelver, modeltp, cls_num):
    model = enet.from_pretrained(f'efficientnet-{modelver}')
    if modeltp == "Original":
        model = model.cuda()
    
    elif modeltp == "RT-FP16":
        model = build_onnx_trt(model, modelver, half=True, n_cls=cls_num)
    
    elif modeltp == "RT-FP32":
        model = build_onnx_trt(model, modelver, half=False, n_cls=cls_num)
    
    else:
        AssertionError("Model Type Error Please Enter Correct Type: Original, RT-FP16, RT-FP32")

    return model 

#@nvtx.annotate("building onnx trt", color='blue')
def build_onnx_trt(model, modelversion, half, n_cls):
    model_path = os.path.abspath('models')
    file_list = os.listdir(model_path)
    onnx_filename = f"{model_path}/efnet-{modelversion}.onnx"

    # Build Pytorch Model -> Onnx Model
    if not any (files.endswith(f"{modelversion}.onnx") for files in file_list):
        print("Building ONNX & TensorRT model\nThis will take about 10 min or more")
        dummy_input = torch.randn(1, 3, 224, 224)
        model.set_swish(memory_efficient=False)
        torch.onnx.export(model, dummy_input, onnx_filename, verbose=True)
    
    # Build Onnx Model -> TRT Model
    if half == True:
        trt_filename = f'{model_path}/trt_{modelversion}_fp16.engine'
        if not any (files.endswith(f"{modelversion}_fp16.engine") for files in file_list):
            convert_trt(onnx_filename, trt_filename, half)
    else:
        trt_filename = f'{model_path}/trt_{modelversion}_fp32.engine'
        if not any (files.endswith(f"{modelversion}_fp32.engine") for files in file_list):
            convert_trt(onnx_filename, trt_filename, half)

    batch_size = 1
    model = ONNXClassifierWrapper(trt_filename,[batch_size, n_cls],target_dtype=np.float32)

    return model

#@nvtx.annotate("img preprocess", color='blue')
def image_preprocess(img):
    image_size=224
    img_trans=tf.Compose([tf.Resize(image_size), tf.CenterCrop(image_size),
                 tf.ToTensor(), tf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    input_img = img_trans(img).unsqueeze(0)
    
    return input_img

#@nvtx.annotate("evaluation", color='blue')
def eval(model, modeltp, input_img):
    if modeltp == 'Original':
        model.eval()
        with torch.no_grad():
            startt = time.time()
            # with profiler.profile(with_stack=True, use_cuda=True, profile_memory=True) as prof:
            #     results = model(input_img.cuda())
            # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
            #with nvtx.annotate("Model Inference", color='green'):
            results = model(input_img.cuda())
            endt = time.time()
    else:
        startt = time.time()
        results=model.predict(input_img.numpy().transpose((0,3,2,1)))
        endt = time.time()
        results=torch.from_numpy(results)

    inft = endt-startt

    return inft, results

def visualize(image, imagename, prediction):
    preds = torch.topk(prediction, k=3).indices.squeeze(0).tolist()
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontscale = 0.5
    color=(0, 255, 0)
    thick = 1
    for cnt, idx in enumerate(preds):
        label = labels_map[idx]
        prob = torch.softmax(results, dim=1)[0, idx]
        text = f'{label} ({(prob*100):.2f}%)'
        position = (5,20 + cnt*20)
        cv2.putText(image, text, position, font, fontscale, color, thick, cv2.LINE_AA)
        print(f'{label:<75} ({(prob*100):.2f}%)')
    cv2.imwrite(f'class_{imagename}', image)
    
if __name__ == "__main__":
    print("""\nEfficientNet Classification Demo\n
          This model will claasify your input image among 1000 classes""")
    current_dir = os.path.abspath('')
          
    modelver = input("\nEnter Model Version to use (b0~b7): ")
    modeltp = input("\nEnter Model type (Originial, RT-FP16, RT-FP32): ")
    imagename = input("\nEnter Image Name to Classify: ")

    labels_map=json.load(open(f'{current_dir}/labels_map.txt'))
    labels_map=[labels_map[str(i)] for i in range(1000)]
    ori_img=Image.open(f"{current_dir}/{imagename}")

    model = build_model(modelver, modeltp, cls_num=len(labels_map))
    img = image_preprocess(ori_img)
    inf_time, results = eval(model, modeltp, img)
    ori_img = cv2.imread(f"{current_dir}/{imagename}")
    visualize(ori_img, imagename, results)
    print(f"\nModel Inference Time: {inf_time*1000:.2f}ms\n")