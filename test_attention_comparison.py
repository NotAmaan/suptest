#!/usr/bin/env python3
"""
Test script to compare SUPIR output quality between different attention mechanisms.
This tests SDPA (current) vs xFormers (old) to isolate the impact of attention implementation.
"""

import torch
import argparse
import os
import time
from PIL import Image
from SUPIR.util import create_SUPIR_model, PIL2Tensor, Tensor2PIL, convert_dtype
import yaml
import shutil

def update_yaml_attention(yaml_path, attention_type):
    """Update the attention type in the YAML config"""
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update attention type in model params
    if 'model' in config and 'params' in config['model']:
        if 'spatial_transformer_attn_type' not in config['model']['params']:
            config['model']['params']['spatial_transformer_attn_type'] = attention_type
        else:
            config['model']['params']['spatial_transformer_attn_type'] = attention_type
    
    # Save to a temporary config file
    temp_yaml = yaml_path.replace('.yaml', f'_{attention_type}.yaml')
    with open(temp_yaml, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    return temp_yaml

def test_attention_availability():
    """Check which attention mechanisms are available"""
    available = []
    
    # Check SDPA
    try:
        if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            available.append('sdpa')
    except:
        pass
    
    # Check xFormers
    try:
        import xformers
        import xformers.ops
        available.append('softmax-xformers')
    except ImportError:
        print("xFormers not available. Install with: pip install xformers")
    
    # Always available
    available.append('softmax')  # vanilla attention
    
    return available

def process_with_attention(input_image_path, attention_type, args):
    """Process an image with a specific attention mechanism"""
    
    print(f"\n{'='*60}")
    print(f"Processing with {attention_type} attention")
    print(f"{'='*60}")
    
    # Update config to use the specified attention type
    config_path = "options/SUPIR_v0_tiled.yaml" if args.use_tile else "options/SUPIR_v0.yaml"
    temp_config = update_yaml_attention(config_path, attention_type)
    
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
        output_dir = os.path.join(args.output_dir, attention_type.replace('-', '_'))
        os.makedirs(output_dir, exist_ok=True)
        
        for i, sample in enumerate(samples):
            output_path = os.path.join(output_dir, f"result_{args.SUPIR_sign}_{args.steps}steps.png")
            Tensor2PIL(sample, h0, w0).save(output_path)
            print(f"Saved: {output_path}")
        
        print(f"Processing time: {processing_time:.2f} seconds")
        print(f"Peak VRAM usage: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
        
        # Clean up
        del model
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"Error with {attention_type}: {e}")
        processing_time = None
    finally:
        # Remove temporary config
        if os.path.exists(temp_config):
            os.remove(temp_config)
    
    return processing_time

def main():
    parser = argparse.ArgumentParser(description="Compare SUPIR quality with different attention mechanisms")
    parser.add_argument("--input", type=str, required=True, help="Input image path")
    parser.add_argument("--output_dir", type=str, default="attention_comparison", help="Output directory")
    parser.add_argument("--SUPIR_sign", type=str, default="Q", choices=["Q", "F"])
    parser.add_argument("--upscale", type=int, default=2)
    parser.add_argument("--steps", type=int, default=50, help="Number of diffusion steps")
    parser.add_argument("--seed", type=int, default=1234567891)
    parser.add_argument("--use_tile", action="store_true", help="Use tiled processing")
    parser.add_argument("--loading_half_params", action="store_true", default=True)
    parser.add_argument("--skip_denoise", action="store_true", help="Skip denoise stage")
    
    # Use old config defaults for Q model
    parser.add_argument("--cfg_start", type=float, default=4.0)
    parser.add_argument("--cfg_end", type=float, default=7.5)
    parser.add_argument("--control_start", type=float, default=1.0)
    parser.add_argument("--control_end", type=float, default=1.0)
    
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
    
    # Check available attention mechanisms
    available_attention = test_attention_availability()
    print(f"Available attention mechanisms: {available_attention}")
    
    # Select which ones to test
    attention_to_test = []
    if 'sdpa' in available_attention:
        attention_to_test.append('sdpa')  # Current default
    if 'softmax-xformers' in available_attention:
        attention_to_test.append('softmax-xformers')  # Old default
    else:
        print("\nWarning: xFormers not available. To test old behavior, install with:")
        print("pip install xformers")
    
    if len(attention_to_test) < 2:
        print("\nNote: Only testing with available attention mechanism(s).")
        print("For a complete comparison, ensure both SDPA and xFormers are available.")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Copy input image to results folder for reference
    input_name = os.path.basename(args.input)
    shutil.copy2(args.input, os.path.join(args.output_dir, f"input_{input_name}"))
    
    # Test each attention mechanism
    results = {}
    for attention_type in attention_to_test:
        processing_time = process_with_attention(args.input, attention_type, args)
        results[attention_type] = processing_time
    
    # Summary
    print(f"\n{'='*60}")
    print("ATTENTION MECHANISM COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"Input: {args.input}")
    print(f"Upscale: {args.upscale}x")
    print(f"SUPIR Model: {args.SUPIR_sign}")
    print(f"Steps: {args.steps}")
    print(f"CFG Scale: {args.cfg_start} -> {args.cfg_end}")
    print(f"Control Scale: {args.control_start} -> {args.control_end}")
    print(f"\nProcessing Times:")
    for attention_type, time_taken in results.items():
        if time_taken:
            print(f"  {attention_type}: {time_taken:.2f} seconds")
        else:
            print(f"  {attention_type}: Failed")
    print(f"\nResults saved to: {args.output_dir}")
    print("\nCompare the output images to see quality differences between attention mechanisms.")
    
    if 'softmax-xformers' not in attention_to_test:
        print("\nNote: To fully replicate old SUPIR behavior, install xFormers and run again.")

if __name__ == "__main__":
    main()