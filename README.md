# 💦 SLURPP Single-step Latent Underwater Restoration - Memory Optimized Fork by CoMPAS Lab at MBARI

[![Original Website](https://img.shields.io/badge/%F0%9F%A4%8D%20Project%20-Website-blue)](https://tianfwang.github.io/slurpp/)
[![Original Paper](doc/badges/badge-pdf.svg)](https://ieeexplore.ieee.org/document/11127006)
[![Original Model](https://img.shields.io/badge/%F0%9F%A4%97-Model-yellow)](https://huggingface.co/Tianfwang/SLURPP)
[![License](https://img.shields.io/badge/License-Apache--2.0-929292)](https://www.apache.org/licenses/LICENSE-2.0)

This repository is a **memory-optimized fork** of the official SLURPP implementation, featuring comprehensive GPU memory optimizations and quality-of-life improvements for inference.

## 🚀 New Features in This Fork

### 🔋 Memory Optimizations

- **Dynamic component loading**: VAE, UNet, and text encoder moved to GPU only when needed
- **Aggressive CPU offloading**: Reduces GPU memory from 11GB → 1-2GB
- **Half precision support**: Optional FP16 mode for further memory savings
- **Selective model loading**: Disable optional components to save memory
- **Automatic resolution scaling**: Reduces input resolution in low memory mode

### 🛠️ Bug Fixes & Improvements

- Fixed tensor dtype mismatches between CPU/GPU
- Resolved device assignment issues in dynamic loading
- Fixed text encoder position_ids dtype errors
- Added original image size restoration after inference
- Optimized GPU-based image resizing
- Better error handling and memory cleanup

### 📊 Memory Usage Comparison

| Mode | Memory Usage | Speed | Quality | Command |
|------|--------------|-------|---------|---------|
| **Original** | 8-11GB | Fast | Best | Default command |
| **Low Memory** | 3-5GB | Medium | Good | `--low_memory` |
| **Ultra Low** | 2-3GB | Slower | Good | `--low_memory --offload_unet` |
| **Minimal** | 1-2GB | Slowest | Reduced | `--low_memory --offload_unet --disable_text_encoder --disable_stage2` |

## 📋 Quick Start

### Installation

```bash
# Clone this memory-optimized fork
git clone https://github.com/CoMPASLab/SLURPP
cd SLURPP

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Download Models

```bash
# Make sure virtual environment is activated
source .venv/bin/activate
python scripts/download_models.py
```

### Memory-Optimized Inference Examples

#### 🟢 Basic Mode (6-8GB)

```bash
# Using the provided script (automatically uses venv)
bash scripts/inference/infer_real.sh
```

#### 🟡 Low Memory Mode (3-5GB)

```bash
# Make sure virtual environment is activated
source .venv/bin/activate
python slurpp/infer_real.py --config slurpp/config/default.yaml \
    --data_dir test_data --output_dir outputs \
    --low_memory
```

#### 🟠 Ultra Low Memory Mode (2-3GB)

```bash
# Make sure virtual environment is activated
source .venv/bin/activate
python slurpp/infer_real.py --config slurpp/config/default.yaml \
    --data_dir test_data --output_dir outputs \
    --low_memory --offload_unet
```

#### 🔴 Minimal Memory Mode (1-2GB)

```bash
# Make sure virtual environment is activated
source .venv/bin/activate
python slurpp/infer_real.py --config slurpp/config/default.yaml \
    --data_dir test_data --output_dir outputs \
    --low_memory --offload_unet --disable_text_encoder --disable_stage2
```

## 🔧 Memory Optimization Options

### Command Line Flags

- `--low_memory`: Enable CPU offloading and reduce resolution to 384px
- `--offload_unet`: Use half precision and gradient checkpointing
- `--disable_text_encoder`: Remove text encoder (unconditional generation)
- `--disable_stage2`: Remove stage2 model (reduces quality but saves memory)
- `--restore_original_size`: Resize output back to original image dimensions

### What Each Option Does

- **`--low_memory`**: Moves VAE & Text Encoder to CPU, reduces inference resolution
- **`--offload_unet`**: Enables FP16 precision + gradient checkpointing
- **`--disable_text_encoder`**: Removes text encoder completely (saves ~500MB)
- **`--disable_stage2`**: Removes stage2 model completely (saves ~1GB)
- **`--restore_original_size`**: GPU-optimized resizing back to original dimensions

### Trade-offs

⚠️ **Performance vs Memory Trade-offs:**

- Lower memory = slower inference (more CPU↔GPU transfers)
- Half precision may slightly reduce quality
- Removing stage2 reduces output quality
- Lower resolution reduces detail
- GPU resizing adds ~1-2s but maintains quality

## 🔍 Technical Details

### Memory Optimization Techniques

1. **Dynamic Component Loading**: Models moved to GPU only during their forward pass
2. **Device Consistency Fixes**: Ensures all tensors match model device/dtype
3. **Memory Cache Management**: Aggressive `torch.cuda.empty_cache()` calls
4. **Selective Loading**: Optional models can be completely disabled
5. **Resolution Scaling**: Input resolution automatically reduced in low memory mode

### Bug Fixes Implemented

- Fixed `RuntimeError: Input type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor) should be the same`
- Fixed `RuntimeError: Expected tensor for argument #1 'indices' to have one of the following scalar types: Long, Int`
- Fixed `AttributeError: 'NoneType' object has no attribute 'device'` when text encoder disabled
- Fixed device mismatch errors in tensor concatenation operations
- Fixed half precision tensor saving issues

## Training (Original Implementation)

### Step 0: Training Data

Our method accepts any folder containing terrestrial image, along with the generated metric depth map from [Depth-Pro](https://github.com/apple/ml-depth-pro). The image and depth map should be in the same folder. The depth map should have the same file name as the image with the ```depth_pro``` suffix. In training scripts set SCRATCH_DATA_DIR location for training data. Please inspect ```slurpp/datasets/UR_revised_dataloader.py``` for more details.

### Step 1: Diffusion Fine Tuning

diffusion unet training script is in  ```scripts/training/learn.sh``` \
run ```scripts/training/learn.sh <NAME_OF_YAML_CONFIG_FILE>```

### Step 2: Cross-Latent Decoder

first run ```scripts/training/infer_stage2.sh``` to generate pairs of diffusion output/gt data \
then run ```scripts/training/stage2.sh``` to train cross-latent decoder using paired data

## Original Authors & Citation

**Original Team**: [Jiayi Wu](https://jiayi-wu-leo.github.io/), [Tianfu Wang](https://tianfwang.github.io/), [Md Abu Bakr Siddique](https://www.linkedin.com/in/bbkrsddque/), [Md Jahidul Islam](https://jahid.ece.ufl.edu/), [Cornelia Fermuller](https://users.umiacs.umd.edu/~fermulcm/), [Yiannis Aloimonos](https://robotics.umd.edu/clark/faculty/350/Yiannis-Aloimonos), [Christopher A. Metzler](https://www.cs.umd.edu/people/metzler).

**Original Paper**: "Single-Step Latent Diffusion for Underwater Image Restoration" - [IEEE TPAMI 2025](https://ieeexplore.ieee.org/document/11127006)

```bibtex
@article{wu2025single,
  title={Single-Step Latent Diffusion for Underwater Image Restoration},
  author={Wu, Jiayi and Wang, Tianfu and Siddique, Md Abu Bakr and Islam, Md Jahidul and Fermuller, Cornelia and Aloimonos, Yiannis and Metzler, Christopher A},
  journal={arXiv preprint arXiv:2507.07878},
  year={2025}
}
```

## Acknowledgements

This code is modified from the following papers, we thank the authors for their work:

Wang Tianfu, Mingyang Xie, Haoming Cai, Sachin Shah, and Christopher A. Metzler. "Flash-split: 2d reflection removal with flash cues and latent diffusion separation." CVPR 2025.

Ke Bingxin, Anton Obukhov, Shengyu Huang, Nando Metzger, Rodrigo Caye Daudt, and Konrad Schindler. "Repurposing diffusion-based image generators for monocular depth estimation." CVPR 2024.

**This memory-optimized fork** adds comprehensive GPU memory optimizations, bug fixes, and quality-of-life improvements while maintaining compatibility with the original research work.

## 📄 License

The code and models of this work are licensed under the Apache License, Version 2.0 (as defined in the [LICENSE](LICENSE)).
