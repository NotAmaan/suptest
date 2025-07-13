#!/usr/bin/env python3
"""
Test script to compare SUPIR output quality between different base models.
This helps isolate whether quality differences are due to:
1. Base model change (SDXL vs Juggernaut)
2. Attention mechanism (xFormers vs SDPA)
3. Default parameter differences
"""

import torch
import argparse
import os
import time
from PIL import Image
from SUPIR.util import create_SUPIR_model, PIL2Tensor, Tensor2PIL, convert_dtype
import shutil
import yaml

def update_yaml_base_model(yaml_path, model_path):
    """Update the SDXL checkpoint path in the YAML config"""
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update the SDXL checkpoint path
    config['SDXL_CKPT'] = model_path
    
    # Save to a temporary config file
    temp_yaml = yaml_path.replace('.yaml', '_temp.yaml')
    with open(temp_yaml, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    return temp_yaml

def process_with_model(input_image_path, base_model_name, base_model_path, args):
    """Process an image with a specific base model"""
    
    print(f"\n{'='*60}")
    print(f"Processing with {base_model_name}")
    print(f"{'='*60}")
    
    # Update config to use the specified base model
    config_path = "options/SUPIR_v0_tiled.yaml" if args.use_tile else "options/SUPIR_v0.yaml"
    temp_config = update_yaml_base_model(config_path, base_model_path)
    
    try:
        # Create SUPIR model
        model = create_SUPIR_model(temp_config, SUPIR_sign=args.SUPIR_sign)
        
        # Configure model
        if args.loading_half_params:
            model = model.half()
        if args.use_tile:
            model.init_tile_vae(encoder_tile_size=1024, decoder_tile_size=256)
        
        model.ae_dtype = convert_dtype('bf16')
        model.model.dtype = convert_dtype('fp16')
        
        # Move to GPU
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        
        # Load and process image
        img = Image.open(input_image_path)
        LQ_img, h0, w0 = PIL2Tensor(img, upscale=args.upscale, min_size=1024)
        LQ_img = LQ_img.unsqueeze(0).to(device)[:, :3, :, :]
        
        # Time the processing
        start_time = time.time()
        
        # Process with SUPIR
        samples = model.batchify_sample(
            LQ_img, 
            args.caption,
            num_steps=args.steps,
            restoration_scale=-1,
            s_churn=5,
            s_noise=1.003,
            cfg_scale_start=args.cfg_start,
            cfg_scale_end=args.cfg_end,
            control_scale_start=args.control_start,
            control_scale_end=args.control_end,
            seed=args.seed,
            num_samples=1,
            p_p=args.positive_prompt,
            n_p=args.negative_prompt,
            color_fix_type='Wavelet',
            skip_denoise_stage=args.skip_denoise
        )
        
        processing_time = time.time() - start_time
        
        # Save result
        output_dir = os.path.join(args.output_dir, base_model_name.replace(' ', '_'))
        os.makedirs(output_dir, exist_ok=True)
        
        for i, sample in enumerate(samples):
            output_path = os.path.join(output_dir, f"result_{args.SUPIR_sign}_{args.steps}steps.png")
            Tensor2PIL(sample, h0, w0).save(output_path)
            print(f"Saved: {output_path}")
        
        print(f"Processing time: {processing_time:.2f} seconds")
        
        # Clean up
        del model
        torch.cuda.empty_cache()
        
    finally:
        # Remove temporary config
        if os.path.exists(temp_config):
            os.remove(temp_config)
    
    return processing_time

def main():
    parser = argparse.ArgumentParser(description="Compare SUPIR quality with different base models")
    parser.add_argument("--input", type=str, required=True, help="Input image path")
    parser.add_argument("--output_dir", type=str, default="comparison_results", help="Output directory")
    parser.add_argument("--SUPIR_sign", type=str, default="Q", choices=["Q", "F"])
    parser.add_argument("--upscale", type=int, default=2)
    parser.add_argument("--steps", type=int, default=50, help="Number of diffusion steps")
    parser.add_argument("--seed", type=int, default=1234567891)
    parser.add_argument("--use_tile", action="store_true", help="Use tiled processing")
    parser.add_argument("--loading_half_params", action="store_true", default=True)
    parser.add_argument("--skip_denoise", action="store_true", help="Skip denoise stage")
    
    # Model-specific default parameters (from old config)
    parser.add_argument("--cfg_start", type=float, help="CFG scale start")
    parser.add_argument("--cfg_end", type=float, help="CFG scale end")
    parser.add_argument("--control_start", type=float, help="Control scale start")
    parser.add_argument("--control_end", type=float, help="Control scale end")
    
    parser.add_argument("--caption", type=str, default="", help="Image caption")
    parser.add_argument("--positive_prompt", type=str, 
                       default="Cinematic, High Contrast, highly detailed, taken using a Canon EOS R camera, "
                               "hyper detailed photo - realistic maximum detail, 32k, Color Grading, ultra HD, "
                               "extreme meticulous detailing, skin pore detailing, hyper sharpness, perfect without deformations.")
    parser.add_argument("--negative_prompt", type=str,
                       default="painting, oil painting, illustration, drawing, art, sketch, oil painting, "
                               "cartoon, CG Style, 3D render, unreal engine, blurring, dirty, messy, "
                               "worst quality, low quality, frames, watermark, signature, jpeg artifacts, "
                               "deformed, lowres, over-smooth")
    
    args = parser.parse_args()
    
    # Set default parameters based on SUPIR_sign if not provided
    if args.cfg_start is None:
        args.cfg_start = 4.0 if args.SUPIR_sign == 'Q' else 1.0
    if args.cfg_end is None:
        args.cfg_end = 7.5 if args.SUPIR_sign == 'Q' else 1.0
    if args.control_start is None:
        args.control_start = 1.0
    if args.control_end is None:
        args.control_end = 1.0
    
    # Define base models to test
    base_models = {
        "SDXL_Original": "models/SDXL/sd_xl_base_1.0.safetensors",  # You'll need to download this
        "Juggernaut_v9": "models/SDXL/juggernautXL_v9Rundiffusionphoto2.safetensors"
    }
    
    # Check if models exist
    models_to_test = {}
    for name, path in base_models.items():
        full_path = os.path.join("/workspace" if os.path.exists("/workspace/models") else ".", path)
        if os.path.exists(full_path):
            models_to_test[name] = full_path
        else:
            print(f"Warning: {name} not found at {full_path}")
    
    if not models_to_test:
        print("Error: No base models found. Please download at least one model.")
        print("\nTo download SDXL base model:")
        print("huggingface-cli download stabilityai/stable-diffusion-xl-base-1.0 sd_xl_base_1.0.safetensors --local-dir models/SDXL/")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Copy input image to results folder for reference
    input_name = os.path.basename(args.input)
    shutil.copy2(args.input, os.path.join(args.output_dir, f"input_{input_name}"))
    
    # Test each model
    results = {}
    for model_name, model_path in models_to_test.items():
        try:
            processing_time = process_with_model(args.input, model_name, model_path, args)
            results[model_name] = processing_time
        except Exception as e:
            print(f"Error processing with {model_name}: {e}")
            results[model_name] = None
    
    # Summary
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"Input: {args.input}")
    print(f"Upscale: {args.upscale}x")
    print(f"SUPIR Model: {args.SUPIR_sign}")
    print(f"Steps: {args.steps}")
    print(f"CFG Scale: {args.cfg_start} -> {args.cfg_end}")
    print(f"Control Scale: {args.control_start} -> {args.control_end}")
    print(f"\nProcessing Times:")
    for model_name, time_taken in results.items():
        if time_taken:
            print(f"  {model_name}: {time_taken:.2f} seconds")
        else:
            print(f"  {model_name}: Failed")
    print(f"\nResults saved to: {args.output_dir}")
    print("\nCompare the output images to see quality differences between base models.")

if __name__ == "__main__":
    main()