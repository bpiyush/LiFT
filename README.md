# LiFT: Linearized Feature Trajectories (NeurIPS 2025)

<p align="center">
  <a href="https://bpiyush.github.io/lift-website/"><img src="https://img.shields.io/badge/🌐-Project_Page-blue?style=plastic" alt="Project Page"></a>
  <a href="https://huggingface.co/datasets/bpiyush/chirality-in-action"><img src="https://img.shields.io/badge/🤗-Dataset-yellow?style=plastic" alt="Dataset"></a>
  <a href="https://neurips.cc/virtual/2025/loc/san-diego/poster/116636"><img src="https://img.shields.io/badge/📄-NeurIPS_2025-red?style=plastic" alt="NeurIPS 2025"></a>
  <a href="https://github.com/bpiyush/LiFT"><img src="https://img.shields.io/badge/💻-GitHub-black?style=plastic" alt="GitHub"></a>
</p>

<h3 align="center">Chirality in Action: Time-Aware Video Representation Learning by Latent Straightening</h3>
<p align="center"><strong>NeurIPS 2025</strong></p>
<p align="center">
  <a href="https://bpiyush.github.io/">Piyush Bagad</a>, &nbsp;
  <a href="https://www.robots.ox.ac.uk/~az/">Andrew Zisserman</a>
</p>
<p align="center">University of Oxford</p>
<img width="1756" height="582" alt="image" src="https://github.com/user-attachments/assets/b725ef05-0e0d-491b-9402-2ad0d01fb1c9" />

---

## Table of Contents

- [Brief Overview](#brief-overview)
- [Installation and Setup](#installation-and-setup)
  - [Download Model Weights](#download-model-weights)
- [Quick Start](#quick-start)
- [Citation](#citation)

---

## Brief Overview

**LiFT** learns time-aware video representations that can linearly separate temporally opposite (chiral) actions like "opening" vs "closing" or "moving up" vs "moving down".

### 🔐 The Key Nugget

**Key observation**: tSNE projections of per-frame features from DINOv2 show that they lie on a time-sensitive trajectory. Can we use these to learn a time-aware video representation?

<p align="center">
<img src="assets/dino_trajectory_example.gif" width="700" alt="DINO Trajectory">
</p>

### 🏗️ The Model: LiFT

**Inspired by perceptual straightening**: LiFT transforms non-linear DINO trajectories into a compact video embedding under a linearized Auto-Encoder model, inspired by the perceptual straightening hypothesis [Hénaff et al., Nature 2019].

<p align="center">
<img src="assets/lift-architecture.gif" width="700" alt="LiFT Architecture">
</p>

<p align="center">
<img src="assets/psh.png" width="600" alt="Perceptual Straightening">
</p>

**What we contribute:**
- **Model**: LiFT - a compact (768-dim) time-aware video embedding trained in an unsupervised manner
- **Benchmark**: Chirality in Action (CiA) - a new benchmark built from SSv2, EPIC, and Charades datasets to evaluate temporal understanding


## Installation and Setup

```sh
pip install git+https://github.com/bpiyush/LiFT.git
```

<details>
<summary><b>Alternative: Manual installation with conda</b></summary>

If you prefer more control over dependencies, create a conda environment:

```sh
conda create --name lift python=3.11 -y
conda activate lift

# Install torch
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# Install lightning
pip install lightning==2.4.0

# Install other dependencies
pip install einops==0.8.1
pip install timm==1.0.22
pip install decord==0.6.0
pip install matplotlib==3.9.2
pip install opencv-python pandas ipdb ipywidgets tqdm scikit-learn termcolor seaborn ffmpeg-python

# Install gdown for downloading model weights
pip install gdown
```

</details>

### Download Model Weights

Download the pre-trained LiFT model weights (~110MB):

```sh
# Download the checkpoint file
gdown 1DFapOrZwRcltyq3_tQNTQ9mHtpgKqtZY -O ggwirp95-epoch=458-step=834003.ckpt
```

Alternatively, you can manually download from [Google Drive](https://drive.google.com/file/d/1DFapOrZwRcltyq3_tQNTQ9mHtpgKqtZY/view?usp=sharing).

## Quick Start

```python
import torch
from lift import DINOv2ForVideo, make_classification_eval_transform, load_lift_module
from lift.dinov2 import compute_dino_features_for_single_video
from lift.demo import compute_lift_embeddings
from lift.viz_utils import show_trajectory_with_reconstruction

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load models
backbone = DINOv2ForVideo(model_id='vit_small_patch14_reg4_dinov2.lvd142m').to(device)
preprocess = make_classification_eval_transform()
lift_model = load_lift_module(ckpt_root=".", ckpt_name="ggwirp95-epoch=458-step=834003.ckpt").to(device)

# Extract features from your video
video_path = "your_video.mp4"
frames, _, dino_feats = compute_dino_features_for_single_video(
    video_path, preprocess, backbone, return_frames=True, device=device, n_frames=16
)

# Get LiFT embedding (768-dim time-aware video representation)
lift_output = compute_lift_embeddings(dino_feats.unsqueeze(0), lift_model, device=device)
embedding = lift_output["concat"]  # Shape: [1, 768]

# Visualize tSNE (DINO trajectory in red, LiFT reconstruction in blue)
img = show_trajectory_with_reconstruction(
    video_path=video_path,
    x=dino_feats,
    x_hat=lift_output["reconstructed"].squeeze(0),
    class_name="my video",
    method="tsne",
    joint_dimred=True,
    return_img=True,
)
img.save("lift_output.png")
```

<img src="lift_output.png" width="500" height="auto" style="display: block; margin: 0 auto;">
<p align="left">
  <em>Visualization of the DINO trajectory (red) and LiFT reconstruction (blue).</em>
</p>

<details>
<summary><b>Alternative: Run the demo script</b></summary>

```sh
cd LiFT
export PYTHONPATH=$PWD
python lift/demo.py --ckpt_root . --ckpt_name ggwirp95-epoch=458-step=834003.ckpt
```

</details>

## Citation

If you find this work useful, please consider citing:

```bibtex
@InProceedings{BagadLiFT25,
  author       = "Piyush Bagad and Andrew Zisserman",
  title        = "Chirality in Action: Time-Aware Video Representation Learning by Latent Straightening",
  booktitle    = "NeurIPS",
  year         = "2025",
}
```

Please also consider checking out the following papers:
* [Seeing the Arrow of Time in Large Multimodal Models](https://vision.cs.utexas.edu/projects/SeeAoT/). NeurIPS (2025).
* [Retro-Actions: Learning ‘Close’ by Time-Reversing ‘Open’ Videos](https://arxiv.org/abs/1909.09422). ICCVW (2019).
* [Perceptual straightening of natural videos](https://www.nature.com/articles/s41593-019-0377-4). Nature Neuroscience (2019).
