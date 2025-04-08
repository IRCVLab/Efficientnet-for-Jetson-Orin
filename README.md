# Classification Demo on Jetson Orin Nano
Load pre-trained [EfficientNet](https://github.com/lukemelas/EfficientNet-PyTorch) model, and make inferences using different engines.

![ex_screenshot](input_img/wolf.jpg.pred_tensorrt_fp16.png)

## Startup
Preparing environment for running EfficientNet in Jetson-Orin
Requirements are listed below
> pytorch 2.5.0
> torchvision 0.20.0
> pycuda 2024.1
> efficientnet_pytorch 0.7.1
> numpy 1.20.3
> onnx 1.16.0
> tensorrt 8.5.2 (SDK Manager will install it)
> cv2 4.5.4 (SDK Manager will install it)

Installation of Pytorch and Torchvision are from [here](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048)

## Install Pytorch and Torchvision

- Install Pytorch for Jetson Nano
```shell
pip3 install http://jetson.webredirect.org/jp6/cu126/+f/5cf/9ed17e35cb752/torch-2.5.0-cp310-cp310-linux_aarch64.whl#sha256=5cf9ed17e35cb7523812aeda9e7d6353c437048c5a6df1dc6617650333049092
```

- Install Torchvision for Jetson Nano
```shell
pip3 install http://jetson.webredirect.org/jp6/cu126/+f/5f9/67f920de3953f/torchvision-0.20.0-cp310-cp310-linux_aarch64.whl#sha256=5f967f920de3953f2a39d95154b1feffd5ccc06b4589e51540dc070021a9adb9
```

- Installation check
```shell
pip3 list
```

**If 'ImportError: libcusparseLt.so.0: cannot open shared object file: No such file or directory' occurs**
```shell
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/arm64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install libcusparselt0 libcusparselt-dev
```

**DO NOT USE 'sudo' command**
- Since we build torchvision from source, if you execute startup.sh with sudo, torchvision will be accessible only with sudo command
```shell
sh startup.sh
```


## TODO
- Install required python packages
    ```bash
    $ pip3 install -r requirements.txt
    ```

- Open the jupyter notebook file (`demo.ipynb`), run each cell, see what happens.
- `classification.py` will do the similar thing at once.
    ```bash
    $ python3 classification.py --engine pytorch
    $ python3 classification.py --engine tensorrt_fp32
    $ python3 classification.py --engine tensorrt_fp16
    ```
    - Use `-h` option to see the usage


## What happens in "Classification.py"
- For the given model version (e.g., `b0`) and inference engine type (e.g., `pytorch`), build the EfficientNet model
    - For more details about Efficientnet, please check [this](https://github.com/lukemelas/EfficientNet-PyTorch) out.
    - If you select TensorRT engine, it will automatically build ONNX & TensorRT model from the original PyTorch model (it takes about 10 minutes)

- Then, open the input image and apply preprocessing
    - Resize to 224 x 224
    - Center crop
    - Convert numpy array (np.ndarray) to torch.Tensor
    - Normalize input: substract mean and divide by std. per channel

- Make a prediction for the input image using the model
- Print inference time
- Overlay predictions on image and save it
