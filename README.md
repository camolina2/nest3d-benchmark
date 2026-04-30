# NEST3D: A High-Resolution Multimodal Dataset of Sociable Weaver Tree Nests

> 🤗 **Dataset:** [huggingface.co/datasets/NEST3D/dataset](https://huggingface.co/datasets/NEST3D/dataset)  
> 📄 **Paper:** [arXiv (coming soon)]()  
> 🐦 **Task:** 3D Point Cloud Semantic Segmentation

---

# NEST3D: A High-Resolution Multimodal Dataset of Sociable Weaver Tree Nests

![NEST3D Workflow](workflow_nest3d.png)

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

## Training Configurations

### PT-v3
- **Optimizer:** AdamW (LR: 0.006, weight decay: 0.05)
- **Scheduler:** OneCycleLR
- **Epochs:** 100
- **Batch size:** 4
- **Loss:** Cross-entropy (nest class weight: 4×) + Lovász loss (weight: 0.5)
- **Voxel grid size:** 0.008
- **Point crop:** 50,000 (random)
- **Input features:** XYZ + RGB (6 channels)

### RandLA-Net
- **Optimizer:** Adam (LR: 0.001)
- **Epochs:** 100
- **Batch size:** 4
- **Layers:** 3 (32 nearest neighbors)
- **Subsampling ratios:** [1, 1, 2]
- **Class-balanced sampling weights:** [1, 3, 1]

### KPConv
- **Optimizer:** Adam (LR: 0.001, weight decay: 0.0005)
- **Epochs:** 200
- **Batch size:** 6
- **Scheduler:** Exponential decay (gamma: 0.99)
- **First subsampling:** 0.01
- **Input radius:** 0.2
- **Convolution radius:** 0.08
- **Kernel points:** 50
- **Layers:** 3

> **Note:** KPConv collapsed to the majority class (Grass) on this dataset, achieving 0% IoU for both Nest and Tree classes. This result highlights the challenges that extreme class imbalance and non-uniform point distributions in ecological data pose for rigid kernel convolution approaches.

---

## Download the Dataset

The dataset is hosted on Hugging Face (~1.4 TB total). Each sample is packaged as a `.tar.gz` archive (~8 GB each). **We strongly recommend downloading one sample at a time** rather than the full dataset.

> 📦 Dataset structure on HuggingFace:
> ```
> train/sample001.tar.gz   # ~8 GB each, 83 training samples
> train/sample002.tar.gz
> ...
> test/sample101.tar.gz    # ~8 GB each, 21 test samples
> ...
> ```

### Option 1: Download a single sample (recommended)

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

### Option 2: Download only the annotated point clouds (lightweight)

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

### Option 3: Download the full dataset (requires ~1.4 TB free disk space)

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

---

## Load and Visualize

### Load a point cloud

Each scene is stored as a `.npy` file with columns `[x, y, z, r, g, b, label]`.

```python
import numpy as np

# Load a training scene
pc = np.load("data/train/sample_012/sample012.npy")

xyz   = pc[:, :3]              # 3D coordinates (meters, local Euclidean frame)
rgb   = pc[:, 3:6]             # RGB color values, scaled to [0, 1]
label = pc[:, 6].astype(int)   # 0=Grass, 1=Nest, 2=Tree

print(f"Points: {len(pc):,} | Classes: {np.unique(label)}")
```

### Visualize a point cloud

```python
import open3d as o3d
import numpy as np

pc = np.load("data/train/sample_012/sample012.npy")
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pc[:, :3])
pcd.colors = o3d.utility.Vector3dVector(pc[:, 3:6])
o3d.visualization.draw_geometries([pcd])
```

### Visualize with semantic labels

```python
import open3d as o3d
import numpy as np

pc = np.load("data/train/sample_012/sample012.npy")
labels = pc[:, 6].astype(int)

# Color map: Grass=green, Nest=red, Tree=brown
color_map = {
    0: [0.18, 0.80, 0.44],  # Grass
    1: [0.91, 0.30, 0.24],  # Nest
    2: [0.55, 0.27, 0.07],  # Tree
}
colors = np.array([color_map[l] for l in labels])

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pc[:, :3])
pcd.colors = o3d.utility.Vector3dVector(colors)
o3d.visualization.draw_geometries([pcd])
```

---

## Models and Implementations

We benchmark three architectures. The implementations and repositories used are:

| Model | Repository | Framework |
|---|---|---|
| PT-v3 | [Pointcept](https://github.com/Pointcept/Pointcept) | PyTorch |
| RandLA-Net | [Open3D-ML](https://github.com/isl-org/Open3D-ML) | PyTorch |
| KPConv | [Open3D-ML](https://github.com/isl-org/Open3D-ML) | PyTorch |

Training configurations and hyperparameters are detailed in the [Training Configurations](#training-configurations) section above. Custom dataset loaders and config files for reproducing our experiments will be released in this repository.

## Model Weights

Pre-trained checkpoints are available on Hugging Face:

| Model | mIoU | Download |
|---|---|---|
| PT-v3 | 86.35% | [NEST3D/nest3d-ptv3](https://huggingface.co/NEST3D/nest3d-ptv3/tree/main) |
| RandLA-Net | 50.72% | [NEST3D/nest3d-randlanet](https://huggingface.co/NEST3D/nest3d-randlanet/tree/main) |
| KPConv | 16.40% | [NEST3D/nest3d-kpconv](https://huggingface.co/NEST3D/nest3d-kpconv/tree/main) |

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
@article{molinacatricheo2026nest3d,
  title={NEST3D: A High-Resolution Multimodal Dataset of Sociable Weaver Tree Nests},
  author={Molina Catricheo, Constanza A. and Boeder, Simon and Guo, Ting-Jia and May, Giacomo and Berthelot, Cl\'{e}ment and Tuia, Devis and Reinhard, Friedrich F. and Remondino, Fabio and Risse, Benjamin},
  journal={arXiv preprint},
  year={2026}
}
```

> 📌 The arXiv ID will be updated once the preprint is published.

---

## Acknowledgements

This work was funded by the European Union's Horizon Europe programme through the Marie Skłodowska-Curie project **WildDrone** (grant no. 101071224), the EPSRC grant **Autonomous Drones for Nature Conservation Missions** (EP/X029077/1), and the Swiss State Secretariat for Education, Research and Innovation (SERI, contract no. 22.00280).

---

## Contact

For questions about the dataset or benchmark, please contact:
- **Constanza A. Molina Catricheo** — cmolinac@uni-muenster.de
- **Benjamin Risse** — b.risse@uni-muenster.de

---

## License

The dataset is released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Code in this repository is released under the [MIT License](LICENSE).