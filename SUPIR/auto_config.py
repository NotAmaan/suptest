"""
Automatic configuration based on available GPU VRAM
"""
import torch
import os

def get_optimal_tile_config():
    """
    Automatically determine optimal tile sizes based on available VRAM
    """
    if not torch.cuda.is_available():
        # CPU fallback
        return {
            'encoder_tile_size': 256,
            'decoder_tile_size': 32,
            'sampler_tile_size': 64,
            'sampler_tile_stride': 32,
            'num_parallel_workers': 1,
            'use_tiled_vae': True
        }
    
    # Get available VRAM in GB
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    # Configuration tiers based on VRAM
    if vram_gb >= 24:  # RTX 4090, 5090, A100, etc.
        config = {
            'encoder_tile_size': 1024,
            'decoder_tile_size': 256,
            'sampler_tile_size': 256,
            'sampler_tile_stride': 128,
            'num_parallel_workers': 6,  # Increased for RTX 5090
            'use_tiled_vae': True
        }
    elif vram_gb >= 16:  # RTX 4080, A4000, etc.
        config = {
            'encoder_tile_size': 768,
            'decoder_tile_size': 128,
            'sampler_tile_size': 192,
            'sampler_tile_stride': 96,
            'num_parallel_workers': 3,
            'use_tiled_vae': True
        }
    elif vram_gb >= 12:  # RTX 4070 Ti, 3080, etc.
        config = {
            'encoder_tile_size': 512,
            'decoder_tile_size': 64,
            'sampler_tile_size': 128,
            'sampler_tile_stride': 64,
            'num_parallel_workers': 2,
            'use_tiled_vae': True
        }
    elif vram_gb >= 8:  # RTX 4060, 3070, etc.
        config = {
            'encoder_tile_size': 384,
            'decoder_tile_size': 48,
            'sampler_tile_size': 96,
            'sampler_tile_stride': 48,
            'num_parallel_workers': 2,
            'use_tiled_vae': True
        }
    else:  # Low VRAM cards
        config = {
            'encoder_tile_size': 256,
            'decoder_tile_size': 32,
            'sampler_tile_size': 64,
            'sampler_tile_stride': 32,
            'num_parallel_workers': 1,
            'use_tiled_vae': True
        }
    
    # Allow environment variable overrides
    if os.environ.get('SUPIR_ENCODER_TILE'):
        config['encoder_tile_size'] = int(os.environ['SUPIR_ENCODER_TILE'])
    if os.environ.get('SUPIR_DECODER_TILE'):
        config['decoder_tile_size'] = int(os.environ['SUPIR_DECODER_TILE'])
    if os.environ.get('SUPIR_WORKERS'):
        config['num_parallel_workers'] = int(os.environ['SUPIR_WORKERS'])
    
    return config

def get_optimal_dtype_config():
    """
    Get optimal dtype configuration based on GPU
    """
    if not torch.cuda.is_available():
        return {
            'ae_dtype': 'fp32',
            'diff_dtype': 'fp32'
        }
    
    # Check GPU compute capability
    major, minor = torch.cuda.get_device_capability()
    
    # Ampere (8.x) and newer have good FP16/BF16 support
    if major >= 8:
        return {
            'ae_dtype': 'bf16',
            'diff_dtype': 'fp16'
        }
    # Turing (7.5) has FP16 but no BF16
    elif major == 7 and minor >= 5:
        return {
            'ae_dtype': 'fp16',
            'diff_dtype': 'fp16'
        }
    else:
        # Older GPUs - use FP32 for stability
        return {
            'ae_dtype': 'fp32',
            'diff_dtype': 'fp32'
        }

def print_config(config, dtype_config):
    """
    Print the configuration being used
    """
    from Y7.colored_print import color
    
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.is_available() else 0
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    
    print(f"\n{'='*60}", color.CYAN)
    print(f"Auto-detected GPU: {gpu_name}", color.CYAN)
    print(f"Available VRAM: {vram_gb:.1f} GB", color.CYAN)
    print(f"{'='*60}", color.CYAN)
    
    print(f"\nTile Configuration:", color.GREEN)
    print(f"  Encoder tile size: {config['encoder_tile_size']}", color.GREEN)
    print(f"  Decoder tile size: {config['decoder_tile_size']}", color.GREEN)
    print(f"  Sampler tile size: {config['sampler_tile_size']}", color.GREEN)
    print(f"  Sampler stride: {config['sampler_tile_stride']}", color.GREEN)
    print(f"  Parallel workers: {config['num_parallel_workers']}", color.GREEN)
    
    print(f"\nDatatype Configuration:", color.YELLOW)
    print(f"  AE dtype: {dtype_config['ae_dtype']}", color.YELLOW)
    print(f"  Diffusion dtype: {dtype_config['diff_dtype']}", color.YELLOW)
    print(f"{'='*60}\n", color.CYAN)