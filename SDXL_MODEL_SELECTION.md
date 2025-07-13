# SDXL Model Selection Feature

This document describes the new SDXL model selection functionality added to SUPIR-Demo.

## Overview

SUPIR now supports switching between different SDXL base models at runtime, allowing you to experiment with different artistic styles and qualities without modifying configuration files.

## Usage

### Command Line Interface

List available SDXL models:
```bash
python3 run_supir_cli.py --list_sdxl_models
```

Use a specific SDXL model:
```bash
python3 run_supir_cli.py --img_path input/image.jpg --sdxl_model "juggernautXL_v9Rundiffusionphoto2.safetensors"
```

### Gradio Web Interface

1. Launch the Gradio interface:
   ```bash
   python3 run_supir_gradio.py
   ```

2. In Tab 2 (Process), find the **"SDXL Base Model"** dropdown
3. Select from:
   - `Default (from config)` - Uses the model specified in YAML config
   - Any `.safetensors` files found in `models/SDXL/`

## Adding New SDXL Models

1. Download SDXL-compatible models (`.safetensors` format)
2. Place them in the `models/SDXL/` directory
3. They will automatically appear in the dropdown/CLI options

## Technical Details

### Implementation

The feature adds:
- `custom_sdxl_path` parameter to `create_SUPIR_model()` function
- `get_available_sdxl_models()` utility function to scan for models
- Runtime model path override capability
- Model caching to avoid unnecessary reloads

### Model Compatibility

- Only SDXL-based models are supported (not SD 1.5 or other architectures)
- Models must be in `.safetensors` format
- The model architecture must match SDXL specifications

### Performance Notes

- Switching models requires reloading the entire SUPIR model
- Model switching is cached - same settings won't trigger reload
- First load may take 30-60 seconds depending on model size

## Default Configuration

To set a default SDXL model in `defaults.json`:

```json
{
  "supir_settings": {
    "sdxl_model": "your_preferred_model.safetensors"
  }
}
```

Use `"Default (from config)"` to use the YAML-specified model.