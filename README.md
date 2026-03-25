# NEST3D: A High-Resolution Multimodal Dataset of Sociable Weaver Tree Nests

> 📄 **Paper under review** — KDD 2026  
> 🤗 **Dataset:** [huggingface.co/datasets/NEST3D/dataset](https://huggingface.co/datasets/NEST3D/dataset)  
> 🐦 **Task:** 3D Point Cloud Semantic Segmentation

---

## Overview

**NEST3D** is an open-access, 1.4 TB multimodal drone dataset of sociable weaver (*Philetairus socius*) nest-bearing trees, collected at the Kuzikus Wildlife Reserve, Namibia. It is the first publicly available dataset to provide expert-annotated 3D point clouds specifically targeting the fine-grained structural segmentation of animal-built nests embedded in complex tree canopies.

The dataset bridges computational ecology and computer vision, enabling applications from automated nest volume estimation to species habitat monitoring.

---

## Dataset at a Glance

| Property | Value |
|---|---|
| Total scenes | 104 trees (83 train / 21 test) |
| Total 3D points | ~781 million |
| RGB images | 27,945 |
| Multispectral images | 111,780 (4 bands: Green, Red, Red Edge, NIR) |
| Semantic classes | Tree · Nest · Grass |
| Point format | `[x, y, z, r, g, b, label]` (`.npy`) |
| Class distribution (train) | Grass 61.7% · Tree 33.9% · **Nest 4.4%** |
| Total size | ~1.4 TB |

> ⚠️ The Nest class represents less than 5% of points, making this a **challenging class-imbalanced benchmark**.

---

## Benchmark Results

We evaluate three state-of-the-art 3D semantic segmentation architectures on the test split. All models use `XYZ + RGB` as input, with unit-sphere normalization.

| Method | Grass IoU | Nest IoU | Tree IoU | **mIoU** | OA |
|---|---|---|---|---|---|
| **PT-v3** | **96.59** | **69.99** | **92.47** | **86.35** | **96.42** |
| RandLA-Net | 80.86 | 17.98 | 53.30 | 50.72 | 73.64 |
| KPConv | 49.19 | 0.00 | 0.00 | 16.40 | 49.19 |

**Key findings:**
- **PT-v3** achieves the best results overall, leveraging transformer-based attention to handle class imbalance and complex nest geometries.
- **RandLA-Net** captures most Nest points but suffers from high false positives (34.1% of Tree points misclassified as Nest).
- **KPConv** collapses to the majority class (Grass), highlighting the sensitivity of convolution-based kernels to sparse, ecologically imbalanced data.

---

## Repository Structure

```
NEST3D/
├── README.md
├── LICENSE
├── environment.yml              # Conda environment for reproducibility
│
├── configs/                     # Training configuration files
│   ├── ptv3_nest3d.py           # PT-v3 config
│   ├── randlanet_nest3d.yml     # RandLA-Net config
│   └── kpconv_nest3d.yml        # KPConv config
│
├── scripts/                     # Training & evaluation scripts
│   ├── train_ptv3.py
│   ├── train_randlanet.py
│   ├── train_kpconv.py
│   └── evaluate.py
│
├── checkpoints/
│   └── README.md                # Links to trained model weights (HuggingFace)
│
└── results/
    └── metrics.csv              # Full per-class and per-scene results
```

---

## Getting Started

### 1. Clone this repository

```bash
git clone https://github.com/camolina2/nest3d-benchmark
cd NEST3D
```

### 2. Set up the environment

```bash
conda env create -f environment.yml
conda activate nest3d
```

### 3. Download the dataset

The dataset is hosted on Hugging Face (~1.4 TB total). Each sample is packaged as a `.tar.gz` archive (~8 GB each). **We strongly recommend downloading one sample at a time** rather than the full dataset.

> 📦 Dataset structure on HuggingFace:
> ```
> train/sample001.tar.gz   # ~8 GB each, 83 training samples
> train/sample002.tar.gz
> ...
> test/sample101.tar.gz    # ~8 GB each, 21 test samples
> ...
> ```

#### Option 1: Download a single sample (recommended)

```python
from huggingface_hub import hf_hub_download
import tarfile, os

# Download one sample archive
hf_hub_download(
    repo_id="NEST3D/dataset",
    repo_type="dataset",
    filename="train/sample001.tar.gz",  # change sample number as needed
    local_dir="./data"
)

# Extract it
with tarfile.open("./data/train/sample001.tar.gz", "r:gz") as tar:
    tar.extractall(path="./data/train/sample001/")
```

#### Option 2: Download only the annotated point clouds (lightweight)

If you only need the 3D point clouds for training (no images), download individual `.npy` files — these are much smaller than the full archives:

```python
from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id="NEST3D/dataset",
    repo_type="dataset",
    filename="train/sample001/sample001.npy",  # ~50-200 MB per file
    local_dir="./data"
)
```

#### Option 3: Download the full dataset (requires ~1.4 TB free disk space)

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="NEST3D/dataset",
    repo_type="dataset",
    local_dir="./data"
)
```

Or browse and download individual files directly at:  
👉 [https://huggingface.co/datasets/NEST3D/dataset](https://huggingface.co/datasets/NEST3D/dataset)

### 4. Load a point cloud

Each scene is stored as a `.npy` file with columns `[x, y, z, r, g, b, label]`.

```python
import numpy as np

# Load a training scene
pc = np.load("data/train/sample_012/sample012.npy")

xyz   = pc[:, :3]          # 3D coordinates (meters, local Euclidean frame)
rgb   = pc[:, 3:6]         # RGB color values, scaled to [0, 1]
label = pc[:, 6].astype(int)  # 0=Grass, 1=Nest, 2=Tree

print(f"Points: {len(pc):,} | Classes: {np.unique(label)}")
```

---

## Reproducing Our Results

### Point Transformer V3 (PT-v3)

We use the [official PT-v3 implementation](https://github.com/Pointcept/PointTransformerV3).

```bash
# Clone PT-v3
git clone https://github.com/Pointcept/PointTransformerV3.git
cd PointTransformerV3

# Copy our config
cp ../configs/ptv3_nest3d.py configs/

# Train
python tools/train.py --config configs/ptv3_nest3d.py

# Evaluate
python tools/test.py --config configs/ptv3_nest3d.py --checkpoint <path_to_checkpoint>
```

### RandLA-Net & KPConv

We use the [Open3D-ML implementations](https://github.com/isl-org/Open3D-ML).

```bash
# Clone Open3D-ML
git clone https://github.com/isl-org/Open3D-ML.git
cd Open3D-ML

# RandLA-Net
python scripts/run_pipeline.py torch -c ../configs/randlanet_nest3d.yml --split train
python scripts/run_pipeline.py torch -c ../configs/randlanet_nest3d.yml --split test

# KPConv
python scripts/run_pipeline.py torch -c ../configs/kpconv_nest3d.yml --split train
python scripts/run_pipeline.py torch -c ../configs/kpconv_nest3d.yml --split test
```

---

## Trained Model Weights

Pre-trained checkpoints are available for download on Hugging Face:

| Model | mIoU | Download |
|---|---|---|
| PT-v3 | 86.35% | 🤗 [nest3d-ptv3.pth](https://huggingface.co/NEST3D) *(coming soon)* |
| RandLA-Net | 50.72% | 🤗 [nest3d-randlanet.pth](https://huggingface.co/NEST3D) *(coming soon)* |
| KPConv | 16.40% | 🤗 [nest3d-kpconv.pth](https://huggingface.co/NEST3D) *(coming soon)* |

---

## Data Collection

- **Location:** Kuzikus Wildlife Reserve, Namibia (Kalahari Desert, ~23°S, 18°E)
- **Drone:** DJI Mavic 3 Multispectral
- **RGB sensor:** 20 MP, 4/3 CMOS
- **Multispectral sensor:** 4×5 MP (Green 560nm, Red 650nm, Red Edge 730nm, NIR 860nm)
- **Collection period:** May 30 – June 22, 2025
- **3D reconstruction:** Agisoft Metashape (photogrammetry from RGB imagery)
- **Annotation tool:** CloudCompare (manual expert labeling)

---

## Ethics and Permissions

Data collection was conducted with official permission from Kuzikus Wildlife Reserve, in accordance with local regulations and institutional guidelines for drone-based ecological fieldwork. No human subjects appear in the dataset except for the authors, who provided explicit consent. Special care was taken to avoid capturing imagery of black rhinos in the reserve.

---

## Citation

If you use NEST3D in your research, please cite:

```bibtex
@inproceedings{molinacatricheo2026nest3d,
  title     = {NEST3D: A High-Resolution Multimodal Dataset of Sociable Weaver Tree Nests},
  author    = {Molina Catricheo, Constanza A. and Boeder, Simon and Guo, Ting-Jia and
               May, Giacomo and Berthelot, Cl{\'e}ment and Tuia, Devis and
               Reinhard, Friedrich F. and Remondino, Fabio and Risse, Benjamin},
  booktitle = {Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  year      = {2026},
  address   = {Jeju, South Korea}
}
```

---

## Acknowledgements

This work was funded by the European Union's Horizon Europe programme through the Marie Skłodowska-Curie project **WildDrone** (grant no. 101071224), the EPSRC grant **Autonomous Drones for Nature Conservation Missions** (EP/X029077/1), and the Swiss State Secretariat for Education, Research and Innovation (SERI, contract no. 22.00280).

---

## License

The dataset is released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Code in this repository is released under the [MIT License](LICENSE).