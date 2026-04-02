# Learning Cross-View Geometric Consistency for Industrial Anomaly Detection
Official implementation of the paper "Learning Cross-View Geometric Consistency for Industrial Anomaly Detection" (Submitted to EAAI 2026).


## 🔨 Requirement
```bash
conda create -n GSANet python==3.8.19
conda activate GSANet
pip install -r requirements.txt
```

## 🐳 Data

#### Real-IAD

Download the Real-IAD dataset from [Real-IAD](https://realiad4ad.github.io/Real-IAD/). Unzip the file and move them to `data/realiad`.

#### Dataset Preprocess

We extract depth maps and camera parametersusing the [VGGT](https://vgg-t.github.io/) method. Run the VGGT preprocessing script on the raw Real-IAD dataset and organize the generated depth maps and camera parameters into `data/depth` and `data/camera`.

## 🚀 Training

```bash
CUDA_VISIBLE_DEVICES=0 python run.py -c configs/gsanet.py -m train
```

## 🍔 Test

```bash
CUDA_VISIBLE_DEVICES=0 python run.py -c configs/gsanet.py -m test 
```

## ❤️ Acknowledgements
Our work is based on [ADer](https://github.com/zhangzjn/ADer), [VGGT](https://vgg-t.github.io/) and [GTA](https://github.com/autonomousvision/gta). Thank them for their excellent works.