from efficientnet_pytorch import EfficientNet as enet
from onnx_trt import ONNXClassifierWrapper, convert_trt
from PIL import Image
from torchvision import transforms as tf

import argparse
import cv2
import json
import numpy as np
import os
import time
import torch

# import torch.autograd.profiler as profiler
# import nvtx

EFNET_MODELS = ('b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7')
ENGINES = ('pytorch', 'tensorrt_fp32', 'tensorrt_fp16')

#@nvtx.annotate("building model", color='blue')
def build_model(modelver, engine, cls_num):
    model = enet.from_pretrained(f'efficientnet-{modelver}')
    if engine == "pytorch":
        model = model.cuda()

    elif engine == "tensorrt_fp16":
        model = build_onnx_trt(model, modelver, half=True, n_cls=cls_num)

    elif engine == "tensorrt_fp32":
        model = build_onnx_trt(model, modelver, half=False, n_cls=cls_num)

    else:
        raise AssertionError(f"Unknown engine: {engine}")

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
    image_size = 224
    img_trans = tf.Compose([
        tf.Resize(image_size),
        tf.CenterCrop(image_size),
        tf.ToTensor(),
        tf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    input_img = img_trans(img).unsqueeze(0)

    return input_img

#@nvtx.annotate("evaluation", color='blue')
def eval(model, modeltp, input_img):
    if modeltp == 'pytorch':
        model.eval()
        with torch.no_grad():
            # warmup
            input_img = input_img.cuda()
            dummy = torch.randn_like(input_img).cuda()
            for _ in range(10):
                model(dummy)

            startt = time.time()
            results = model(input_img)
            endt = time.time()

            inft = endt-startt

            # with profiler.profile(with_stack=True, use_cuda=True, profile_memory=True) as prof:
            #     results = model(input_img.cuda())
            # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
            #with nvtx.annotate("Model Inference", color='green'):

    else:
        input_img_trt = input_img.numpy().transpose((0,3,2,1))

        # warmup
        dummy = np.random.randn(*input_img_trt.shape).astype(np.float32)
        for _ in range(10):
            model.predict(dummy)

        inft, results = model.predict(input_img_trt, eval_exec_time=True)
        results = torch.from_numpy(results)

    return inft, results

def visualize(image, filename, prediction, label_map):
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    preds = torch.topk(prediction, k=3).indices.squeeze(0).tolist()
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontscale = 0.5
    color=(0, 255, 0)
    thick = 1
    for cnt, idx in enumerate(preds):
        label = label_map[idx]
        prob = torch.softmax(results, dim=1)[0, idx]
        text = f'{label} ({(prob*100):.2f}%)'
        position = (5,20 + cnt*20)
        cv2.putText(image, text, position, font, fontscale, color, thick, cv2.LINE_AA)
        print(f'{label:<75} ({(prob*100):.2f}%)')
    cv2.imwrite(filename, image)


if __name__ == "__main__":
    args = argparse.ArgumentParser()

    args.add_argument('--model_ver', choices=EFNET_MODELS, type=str.lower, default='b0', help='EfficientNet model version')
    args.add_argument('--engine', choices=ENGINES, type=str.lower, default='pytorch', help='Inference engine')
    args.add_argument('--image_file', type=str, default='input_img/wolf.jpg', help='Inference target image')
    args = args.parse_args()

    assert args.image_file.endswith('.jpg') or args.image_file.endswith('.png')
    assert os.path.exists(args.image_file)

    print("""\n===== EfficientNet Classification Demo =====
          This model will claasify your input image among 1000 classes""")

    with open('labels_map.txt', 'r') as f:
        label_map = {int(k): v for k, v in json.load(f).items()}

    model = build_model(args.model_ver, args.engine, cls_num=len(label_map))

    image = Image.open(args.image_file)
    img = image_preprocess(image)
    inf_time, results = eval(model, args.engine, img)
    print(f"\nModel Inference Time: {inf_time * 1000:.2f}ms\n")

    # Pillow (PIL.Image): RGB -> OpenCV (np.ndarray): BGR
    image_np = np.array(image)[:,:,(2,1,0)]
    image_np = np.ascontiguousarray(image_np)
    filename = args.image_file + f'.pred_{args.engine}.png'
    visualize(image_np, filename, results, label_map)
