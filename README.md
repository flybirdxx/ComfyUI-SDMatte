# ComfyUI-SDMatte

English | [简体中文](README_CN.md)

ComfyUI custom nodes plugin based on [SDMatte](https://github.com/vivoCameraResearch/SDMatte) for interactive image matting.

## 📖 Introduction

SDMatte is an interactive image matting method based on Stable Diffusion, developed by the vivo Camera Research team and accepted by ICCV 2025. This method leverages the powerful priors of pre-trained diffusion models and supports multiple visual prompts (points, boxes, masks) for accurately extracting target objects from natural images.

This plugin integrates SDMatte into ComfyUI, providing a simple and easy-to-use node interface focused on trimap-guided matting functionality with built-in VRAM optimization strategies.

## ✨ Features

- 🎯 **High-Precision Matting**: Based on powerful diffusion model priors, capable of handling complex edge details
- 🖼️ **Trimap Guidance**: Supports trimap-guided precise matting
- 🚀 **VRAM Optimization**: Built-in mixed precision, attention slicing, and other memory optimization strategies
- 🔧 **ComfyUI Integration**: Fully compatible with ComfyUI workflow system
- 📱 **Flexible Sizes**: Supports multiple inference resolutions (512-1024px)

## 🛠️ Installation

### 1. Download Plugin

Place this plugin in the ComfyUI custom nodes directory:

```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/flybirdxx/ComfyUI-SDMatte.git
```

### 2. Install Dependencies

ComfyUI will automatically install the dependencies in `requirements.txt` on startup:

- diffusers
- timm
- einops
- lazyconfig

### 3. Prepare Model Files

#### Download SDMatte Weights

Download SDMatte model weights from [Hugging Face](https://huggingface.co/LongfeiHuang/SDMatte):

**Standard Version (SDMatte):**
```bash
# Download standard version weight file
wget https://huggingface.co/LongfeiHuang/SDMatte/resolve/main/SDMatte.pth
```

**Enhanced Version (SDMatte+):**
```bash
# Download enhanced version weight file (higher accuracy, larger model)
wget https://huggingface.co/LongfeiHuang/SDMatte/resolve/main/SDMatte_plus.pth
```

Place the downloaded weight files in ComfyUI's checkpoints directory:

```
ComfyUI/models/checkpoints/
├── SDMatte.pth          # Standard version
└── SDMatte_plus.pth     # Enhanced version
```

**Model Selection Guide:**
- **SDMatte.pth**: Standard version, balanced performance and quality, recommended for daily use
- **SDMatte_plus.pth**: Enhanced version, higher accuracy but requires more VRAM and computation time, suitable for high-quality demands

### 4. Restart ComfyUI

Restart ComfyUI to load the new custom nodes.

## 🎮 Usage

### Node Description

#### SDMatte Model Loader

- **Function**: Load SDMatte model
- **Input**:
  - `ckpt_name`: Select SDMatte.pth file from checkpoints directory
- **Output**:
  - `SDMATTE_MODEL`: Loaded SDMatte model

#### SDMatte Apply

- **Function**: Apply SDMatte model for matting
- **Input**:
  - `sdmatte_model`: SDMatte model from model loader
  - `image`: Input image (ComfyUI IMAGE format)
  - `trimap`: Trimap mask (ComfyUI MASK format)
  - `inference_size`: Inference resolution (512/640/768/896/1024)
  - `is_transparent`: Whether the image contains transparent areas
  - `force_cpu`: Force CPU inference (optional)
- **Output**:
  - `MASK`: Alpha mask of matting result

### Basic Workflow

1. **Load Image**: Load the image that needs matting
2. **Create Trimap**: Use drawing tools or other nodes to create trimap
   - Black (0): Definite background
   - White (1): Definite foreground  
   - Gray (0.5): Unknown region
3. **SDMatte Model Loader**: Load SDMatte model
4. **SDMatte Apply**: Apply matting
5. **Preview Image**: Preview matting result

### Recommended Settings

- **Inference Resolution**: 1024 (highest quality) or 768 (balanced performance)
- **Transparent Flag**: Set according to whether input image has transparent channel
- **Force CPU**: Use only when GPU VRAM is insufficient

## 🔧 Technical Details

### Data Processing

- **Input Image**: Automatically resized to inference resolution, normalized to [-1, 1]
- **Trimap**: Resized to inference resolution, mapped to [-1, 1] range
- **Output**: Resized back to original resolution, clamped to [0, 1] range

### VRAM Optimization

The plugin has built-in memory optimization strategies (automatically enabled):

- **Mixed Precision**: Uses FP16 autocast to reduce VRAM usage
- **Attention Slicing**: SlicedAttnProcessor(slice_size=1) maximizes VRAM savings
- **Memory Cleanup**: Automatically clears CUDA cache before and after inference
- **Device Management**: Smart device allocation and model movement

### Model Loading

- **Weight Formats**: Supports .pth and .safetensors formats
- **Safe Loading**: Handles omegaconf objects, supports weights_only mode
- **Nested Structure**: Automatically handles complex checkpoint structures
- **Error Recovery**: Multiple fallback mechanisms ensure successful loading

## ❓ FAQ

### Q: Nodes cannot be searched?
A: Ensure the plugin directory structure is correct, restart ComfyUI, check console for error messages.

### Q: Model loading failed?
A: Check SDMatte.pth file path, ensure base model directory structure is complete, view console for detailed error messages.

### Q: Insufficient VRAM during inference?
A: Try reducing inference resolution, enable `force_cpu` option, or close other VRAM-consuming programs.

### Q: Poor matting results?
A: Optimize trimap quality, ensure accurate foreground/background/unknown region annotations, try different inference resolutions.

### Q: First inference is slow?
A: First run needs to compile CUDA kernels, subsequent inference will be significantly faster.

### Q: Which model version should I choose?
A: 
- **SDMatte.pth (Standard)**: Smaller file (~11GB), faster inference, suitable for most scenarios
- **SDMatte_plus.pth (Enhanced)**: Larger file, higher accuracy, suitable for professional use with extremely high quality requirements
- Recommend testing with standard version first, upgrade to enhanced version if higher quality is needed

## 📋 System Requirements

- **ComfyUI**: Latest version
- **Python**: 3.8+
- **PyTorch**: 1.12+ (CUDA support recommended)
- **VRAM**: 8GB+ recommended (CPU inference supported)
- **Dependencies**: diffusers, timm, einops, lazyconfig

## 📚 References

- **Original Paper**: [SDMatte: Grafting Diffusion Models for Interactive Matting](https://arxiv.org/abs/2408.00321) (ICCV 2025)
- **Original Code**: [vivoCameraResearch/SDMatte](https://github.com/vivoCameraResearch/SDMatte)
- **Model Weights**: [LongfeiHuang/SDMatte](https://huggingface.co/LongfeiHuang/SDMatte)

## 📄 License

This project follows the MIT license. The original SDMatte project also uses the MIT license.

## 🙏 Acknowledgements

Thanks to the vivo Camera Research team for developing the excellent SDMatte model, and to the Stable Diffusion and ComfyUI communities for their contributions.

## 📧 Support

If you have any questions or suggestions, please submit an Issue on GitHub.

---

**Note**: This plugin is a third-party implementation and is not directly affiliated with the original SDMatte team. Please ensure compliance with relevant license terms before use.
