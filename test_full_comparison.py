#!/usr/bin/env python3
"""
Comprehensive test to identify quality differences between old and new SUPIR.
Tests multiple variables:
1. Base model (SDXL vs Juggernaut)
2. Attention mechanism (xFormers vs SDPA)
3. Default parameters (old config values vs new)
"""

import torch
import argparse
import os
import time
import json
from PIL import Image
from datetime import datetime

def run_comparison(input_image, output_dir="quality_comparison"):
    """Run a comprehensive comparison test"""
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{output_dir}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Test configurations
    test_configs = [
        {
            "name": "new_defaults",
            "description": "Current SUPIR defaults (Juggernaut + SDPA)",
            "base_model": "juggernautXL_v9Rundiffusionphoto2.safetensors",
            "attention": "sdpa",
            "cfg_start": 2.0,
            "cfg_end": 4.0,
            "control_start": 0.9,
            "control_end": 0.9
        },
        {
            "name": "old_defaults", 
            "description": "Old SUPIR defaults (Juggernaut + SDPA + old params)",
            "base_model": "juggernautXL_v9Rundiffusionphoto2.safetensors",
            "attention": "sdpa",
            "cfg_start": 4.0,  # Old Q model defaults
            "cfg_end": 7.5,
            "control_start": 1.0,
            "control_end": 1.0
        },
        {
            "name": "sdxl_new_params",
            "description": "SDXL base + new parameters",
            "base_model": "sd_xl_base_1.0.safetensors",
            "attention": "sdpa",
            "cfg_start": 2.0,
            "cfg_end": 4.0,
            "control_start": 0.9,
            "control_end": 0.9
        },
        {
            "name": "sdxl_old_params",
            "description": "SDXL base + old parameters",
            "base_model": "sd_xl_base_1.0.safetensors",
            "attention": "sdpa",
            "cfg_start": 4.0,
            "cfg_end": 7.5,
            "control_start": 1.0,
            "control_end": 1.0
        }
    ]
    
    # If xformers is available, add those configs too
    try:
        import xformers
        test_configs.extend([
            {
                "name": "juggernaut_xformers",
                "description": "Juggernaut + xFormers + old params",
                "base_model": "juggernautXL_v9Rundiffusionphoto2.safetensors",
                "attention": "softmax-xformers",
                "cfg_start": 4.0,
                "cfg_end": 7.5,
                "control_start": 1.0,
                "control_end": 1.0
            },
            {
                "name": "sdxl_xformers",
                "description": "SDXL + xFormers + old params (closest to original)",
                "base_model": "sd_xl_base_1.0.safetensors",
                "attention": "softmax-xformers",
                "cfg_start": 4.0,
                "cfg_end": 7.5,
                "control_start": 1.0,
                "control_end": 1.0
            }
        ])
    except ImportError:
        print("xFormers not available - skipping xFormers tests")
    
    # Common parameters
    common_params = {
        "upscale": 2,
        "steps": 50,
        "SUPIR_sign": "Q",
        "seed": 1234567891,
        "use_tile": True,
        "caption": "",
        "positive_prompt": "Cinematic, High Contrast, highly detailed, taken using a Canon EOS R camera, hyper detailed photo - realistic maximum detail, 32k, Color Grading, ultra HD, extreme meticulous detailing, skin pore detailing, hyper sharpness, perfect without deformations.",
        "negative_prompt": "painting, oil painting, illustration, drawing, art, sketch, oil painting, cartoon, CG Style, 3D render, unreal engine, blurring, dirty, messy, worst quality, low quality, frames, watermark, signature, jpeg artifacts, deformed, lowres, over-smooth"
    }
    
    # Write test configuration
    with open(os.path.join(output_dir, "test_config.json"), "w") as f:
        json.dump({
            "input_image": input_image,
            "timestamp": timestamp,
            "common_params": common_params,
            "test_configs": test_configs
        }, f, indent=2)
    
    # Create comparison script calls
    print(f"Test configurations saved to: {output_dir}/test_config.json")
    print("\nTo run the comparison tests, execute these commands:\n")
    
    # Model comparison
    print("# 1. Base Model Comparison (Juggernaut vs SDXL):")
    print(f"python test_model_comparison.py --input '{input_image}' --output_dir '{output_dir}/models' --steps 50 --use_tile")
    
    # Attention comparison  
    print("\n# 2. Attention Mechanism Comparison (SDPA vs xFormers):")
    print(f"python test_attention_comparison.py --input '{input_image}' --output_dir '{output_dir}/attention' --steps 50 --use_tile")
    
    # Parameter comparison using CLI
    print("\n# 3. Parameter Comparison (using CLI for precise control):")
    for config in test_configs[:4]:  # Just the main 4 configs
        print(f"\n# {config['description']}:")
        print(f"python run_supir_cli.py \\")
        print(f"  --img_path '{input_image}' \\")
        print(f"  --save_dir '{output_dir}/params/{config['name']}' \\")
        print(f"  --SUPIR_sign Q --upscale 2 --edm_steps 50 \\")
        print(f"  --cfg_scale_start {config['cfg_start']} --cfg_scale_end {config['cfg_end']} \\")
        print(f"  --control_scale_start {config['control_start']} --control_scale_end {config['control_end']} \\")
        print(f"  --use_tile_vae --loading_half_params")
    
    print(f"\n\nAll results will be saved to: {output_dir}/")
    print("\nAfter running all tests, compare the output images to identify which factor(s) affect quality most.")
    
    return output_dir

def main():
    parser = argparse.ArgumentParser(description="Setup comprehensive SUPIR quality comparison")
    parser.add_argument("--input", type=str, required=True, help="Input image path")
    parser.add_argument("--output_dir", type=str, default="quality_comparison", help="Base output directory")
    
    args = parser.parse_args()
    
    # Check if input exists
    if not os.path.exists(args.input):
        print(f"Error: Input image not found: {args.input}")
        return
    
    # Run comparison setup
    output_dir = run_comparison(args.input, args.output_dir)
    
    # Create a quick comparison script
    quick_script = os.path.join(output_dir, "run_all_tests.sh")
    with open(quick_script, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("# Quick script to run all comparison tests\n\n")
        f.write("echo 'Starting SUPIR quality comparison tests...'\n\n")
        
        f.write("# Model comparison\n")
        f.write(f"python test_model_comparison.py --input '{args.input}' --output_dir '{output_dir}/models' --steps 50 --use_tile\n\n")
        
        f.write("# Attention comparison\n") 
        f.write(f"python test_attention_comparison.py --input '{args.input}' --output_dir '{output_dir}/attention' --steps 50 --use_tile\n\n")
        
        f.write("# Parameter comparisons\n")
        f.write(f"python run_supir_cli.py --img_path '{args.input}' --save_dir '{output_dir}/params/new_defaults' --SUPIR_sign Q --upscale 2 --edm_steps 50 --cfg_scale_start 2.0 --cfg_scale_end 4.0 --control_scale_start 0.9 --control_scale_end 0.9 --use_tile_vae --loading_half_params\n\n")
        
        f.write(f"python run_supir_cli.py --img_path '{args.input}' --save_dir '{output_dir}/params/old_defaults' --SUPIR_sign Q --upscale 2 --edm_steps 50 --cfg_scale_start 4.0 --cfg_scale_end 7.5 --control_scale_start 1.0 --control_scale_end 1.0 --use_tile_vae --loading_half_params\n\n")
        
        f.write("echo 'All tests completed! Check results in: " + output_dir + "'\n")
    
    os.chmod(quick_script, 0o755)
    print(f"\nQuick test script created: {quick_script}")
    print("Run it with: bash " + quick_script)

if __name__ == "__main__":
    main()