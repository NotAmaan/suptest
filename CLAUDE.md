# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SUPIR-Demo is an advanced AI-powered image upscaling and restoration system that uses diffusion models. It's a customized fork of the original SUPIR project with improvements including:
- Safetensors support
- Updated dependencies and better memory management
- Parallel processing for Tiled VAE
- Metadata saving in PNG files
- More intuitive argument naming

The system combines SDXL diffusion models with specialized SUPIR restoration models to enhance low-resolution or degraded images.

## Key Commands

### Installation & Setup
```bash
# Install dependencies (Linux)
./install_linux_local.sh

# Download models
./download_models.sh

# Activate virtual environment before running
source venv/bin/activate
```

### Running the Application
```bash
# Gradio Web Interface (preferred for interactive use)
python3 run_supir_gradio.py
# or
./launch_gradio.sh

# CLI for batch processing
python3 run_supir_cli.py --img_path 'input/image.jpg' --save_dir ./output --SUPIR_sign Q --upscale 2
```

### Testing
No formal test suite exists. Test by running the application with sample images in the `input/` directory.

## Architecture

### Core Workflow
1. **Input Processing**: Image loaded and optionally denoised via VAE encoder
2. **Upscaling**: Image upscaled to target resolution (minimum 1024px)
3. **Latent Space**: VAE encodes image to latent representation
4. **Diffusion**: SDXL + SUPIR models perform restoration
5. **Decoding**: VAE decodes back to image space
6. **Post-processing**: Optional color correction (Wavelet/AdaIn)

### Key Components
- **`run_supir_gradio.py`**: Web interface entry point
- **`run_supir_cli.py`**: CLI entry point
- **`SUPIR/`**: Core restoration model implementations
- **`sgm/`**: Stable diffusion model components
- **`Y7/`**: Model management utilities
- **`options/`**: Configuration files (SUPIR_v0.yaml, SUPIR_v0_tiled.yaml)

### Model Variants
- **SUPIR-v0Q (Quality)**: Robust for heavily degraded images but may over-enhance
- **SUPIR-v0F (Fidelity)**: Better for lightly degraded images, preserves fine details

### Memory Management
- Use `--use_tile_vae` for large images
- Use `--loading_half_params` to reduce VRAM usage
- TiledRestoreEDMSampler is default for memory efficiency

## Important Configuration

### User Settings
Create `defaults.json` from `defaults_example.json` to customize default parameters.

### Model Paths
Models are expected in:
- `models/SUPIR/`: SUPIR restoration models
- `models/SDXL/`: Base diffusion model
- `models/CLIP1/` & `models/CLIP2/`: CLIP encoders
- `models/SmolVLM-500M-Instruct/`: Caption model for Gradio

### Key Parameters
- `upscale`: Factor to upscale (2-4x recommended)
- `SUPIR_sign`: 'Q' for quality, 'F' for fidelity
- `edm_steps`: Diffusion steps (default 50)
- `cfg_scale_start/end`: Prompt guidance strength
- `control_scale_start/end`: Input image structure guidance

## Development Notes

- Always activate the virtual environment before running
- The system requires significant VRAM (12GB+ recommended)
- Processing time scales quadratically with upscale factor
- Image dimensions are snapped to multiples of 64
- Minimum output resolution is 1024px for SDXL compatibility