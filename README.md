# Classification Demo on Jetson Orin Nano

Load a pre-trained [EfficientNet](https://github.com/lukemelas/EfficientNet-PyTorch) model and perform image classification using multiple inference engines (PyTorch FP32, TensorRT FP32, TensorRT FP16).

![example](input_img/wolf.jpg.pred_tensorrt_fp16.png)

---

## 🚀 Overview

| Component | Description |
|------------|-------------|
| Device | NVIDIA Jetson Orin Nano or AGX orin |
| JetPack | v512 (JP 5.1.x, L4T R35.x) |
| Python | 3.8 (aarch64) |
| Frameworks | PyTorch / TensorRT / ONNX / EfficientNet |

This demo:
- Loads a pre-trained EfficientNet model  
- Converts it to ONNX and TensorRT  
- Runs inference and visualizes predictions  

---

## ⚙️ Requirements

> **⚠️ Do not use `sudo pip`** – always install with `--user`.

| Package | Version | Notes |
|----------|----------|--------|
| pytorch | 2.1.0a0+41361538.nv23.06 | NVIDIA JetPack 5.1 wheel |
| torchvision | 0.16.1 | Build from source |
| pycuda | 2024.1 | CUDA Python interface |
| efficientnet-pytorch | 0.7.1 | Model wrapper |
| numpy | ≥1.20,<2 | Stable range |
| onnx | 1.16.0 | Model export |
| tensorrt | 8.5.2 | Installed via SDK Manager |
| cv2 | 4.5.4 | Installed via SDK Manager |

---

## 🔍 Check JetPack Version & Choose Correct PyTorch/torchvision (JP5/JP6)

This project targets **JetPack 5.x (Python 3.8)** by default.  
If you’re unsure which JetPack version your Jetson is running, follow the steps below to **detect your version** and choose the correct **PyTorch + torchvision** pair.  

---

### 0️⃣ Check Your JetPack Version

```bash
cat /etc/nv_tegra_release

# Example output:
# R35 (release), REVISION: 4.1  → JetPack 5.1.2 (v512, L4T R35.4.x)
# R35 (release), REVISION: 2.1  → JetPack 5.1   (v512, L4T R35.2.x)
# R35 (release), REVISION: 1    → JetPack 5.0.x (v502, L4T R35.1.x)
```

| If your `/etc/nv_tegra_release` says…                    | Install this **torch**             | And this **torchvision**                   |
| -------------------------------------------------------- | ---------------------------------- | ------------------------------------------ |
| **R35.2.x / R35.3.x / R35.4.x / R35.6.x** (JP **5.1.x**) | **torch 2.1.0a0 (nv23.06, cp38)**  | **torchvision 0.16.1** (build from source) |
| **R35.1.x** (JP **5.0.x**)                               | **torch 1.13.0a0 (nv22.10, cp38)** | **torchvision 0.14.1** (build from source) |


## 🧱 Installation

### 1️⃣ System Dependencies
```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential cmake git \
  libjpeg-dev zlib1g-dev libpng-dev \
  libavcodec-dev libavformat-dev libswscale-dev \
  libopenblas-dev libpython3-dev
```

Remove legacy matplotlib (prevents version conflicts):

```bash
sudo apt-get purge -y python3-matplotlib || true
```

### 2️⃣ Install PyTorch (NVIDIA wheel)

#### JP 5.1.x (R35.2/3/4/6 → use torch 2.1 + tv 0.16)

``` bash
python3 -m pip install --user --no-cache-dir \
  https://developer.download.nvidia.cn/compute/redist/jp/v512/pytorch/torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl
```

#### JP 5.0.x (R35.1.x → use torch 1.13 + tv 0.14)

```bash
python3 -m pip install --user --no-cache-dir \
  https://developer.download.nvidia.com/compute/redist/jp/v502/pytorch/torch-1.13.0a0+d0d6b1f2.nv22.10-cp38-cp38-linux_aarch64.whl
```

#### JP 6.x.x
> [!NOTE]
> If you are on JetPack 6 (R36.x / Python 3.10), use the JP6 wheels catalog instead:
https://pypi.jetson-ai-lab.io/jp6/  


Verify installation:

```bash
python3 - << 'PY'
import torch
print("torch        :", torch.__version__)
print("cuda avail   :", torch.cuda.is_available())
print("cuda version :", torch.version.cuda)
print("cudnn        :", torch.backends.cudnn.version())
PY
```

> ✅ `cuda avail` **must** be `True`.

### 3️⃣ Build and Install torchvision
> [!NOTE]
> Check the version of `torchvision` that needs to be installed.
```bash
git clone --branch v0.16.1 https://github.com/pytorch/vision.git
#or
# git clone --branch v0.14.1 https://github.com/pytorch/vision.git
cd vision

# For Jetson Orin (SM 8.7)
export TORCH_CUDA_ARCH_LIST="8.7"
export USE_NINJA=1

python3 -m pip install --user --no-build-isolation --no-cache-dir .

cd ..
```

### 4️⃣ Install Python Packages
``` bash
python3 -m pip install --user -r requirements.txt
```

### 5️⃣ Verify Installation
```bash
python3 - << 'PY'
from efficientnet_pytorch import EfficientNet
import torch
m = EfficientNet.from_pretrained('efficientnet-b0')
m = m.cuda()
x = torch.randn(1,3,224,224, device='cuda')
y = m(x)
print("✅ Inference OK:", y.shape)
PY
```

## 🧪 Running the Demo

### Using Jupyter Notebook
```bash
jupyter notebook demo.ipynb
```

### Using Python Script
```bash
python3 classification.py --engine pytorch
python3 classification.py --engine tensorrt_fp32
python3 classification.py --engine tensorrt_fp16
```
Use `-h` to show available options.


## 🧬 What Happens in `classification.py`

1. **Model Build** – Loads EfficientNet (e.g. `b0`) depending on chosen engine  
2. **Preprocessing** –  
   - Resize to `224×224`  
   - Center crop  
   - Convert to tensor  
   - Normalize by channel  
3. **Inference** – Runs model, measures execution time  
4. **Postprocessing** –  
   - Decodes top-5 predictions  
   - Overlays label on image  
   - Saves result as `*.pred_*.png`

---

## 🩺 Troubleshooting

| Problem | Cause | Fix |
|----------|--------|-----|
| `Torch not compiled with CUDA enabled` | Wrong wheel (CPU-only) | Install NVIDIA JetPack wheel |
| `torchvision` import error | Version mismatch | Use `torch==2.1` + `torchvision==0.16` |
| `seaborn/matplotlib rcParams` error | Mixed apt/pip versions | `sudo apt purge python3-matplotlib`, reinstall via pip |
| `ModuleNotFoundError: mplot3d` | Broken mpl_toolkits | Remove old directory and reinstall matplotlib |
| Jupyter import errors | Wrong Python kernel | Use `ipykernel` and select `jetson-user` |

---

## 🧰 Tips

- Never use `sudo pip`.  
- Always verify CUDA availability before running inference.  
- Rebuild TensorRT engines if `.onnx` or `.engine` files are corrupted.  
- To reset the environment:
  ```bash
  rm -rf ~/.local/lib/python3.8/site-packages/mpl_toolkits
  python3 -m pip install --user --force-reinstall matplotlib seaborn

## 📚 References

- [**EfficientNet-PyTorch**](https://github.com/lukemelas/EfficientNet-PyTorch)  
  Official PyTorch implementation of EfficientNet models.

- [**NVIDIA PyTorch for Jetson**](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048)  
  Official NVIDIA developer forum thread providing PyTorch wheels optimized for Jetson devices.

- [**TensorRT Developer Guide**](https://developer.nvidia.com/tensorrt)  
  Comprehensive documentation on building, optimizing, and deploying deep learning inference with TensorRT.
