"""
Optimized model loading with CPU preloading and parallel loading support
"""
import torch
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from Y7.colored_print import color

# Global cache for preloaded models
_model_cache = {}
_cache_lock = threading.Lock()

def preload_model_to_cpu(path, model_name):
    """Preload a model to CPU memory for faster GPU loading later"""
    try:
        print(f"Preloading {model_name} to CPU...", color.CYAN)
        start = time.time()
        
        # Load to CPU
        if path.endswith('.safetensors'):
            from safetensors.torch import load_file
            state_dict = load_file(path, device='cpu')
        else:
            state_dict = torch.load(path, map_location='cpu')
        
        # Cache it
        with _cache_lock:
            _model_cache[model_name] = state_dict
        
        elapsed = time.time() - start
        print(f"✓ {model_name} preloaded to CPU in {elapsed:.1f}s", color.GREEN)
        return model_name, state_dict
    except Exception as e:
        print(f"✗ Failed to preload {model_name}: {e}", color.RED)
        return model_name, None

def preload_all_models_parallel():
    """Preload all models to CPU in parallel"""
    model_paths = {
        'SDXL': os.environ.get('SDXL_CKPT_PATH', '/workspace/models/SDXL/juggernautXL_v9Rundiffusionphoto2.safetensors'),
        'SUPIR_Q': os.environ.get('SUPIR_CKPT_Q_PATH', '/workspace/models/SUPIR/SUPIR-v0Q_fp16.safetensors'),
        'SUPIR_F': os.environ.get('SUPIR_CKPT_F_PATH', '/workspace/models/SUPIR/SUPIR-v0F_fp16.safetensors'),
        'CLIP1': os.environ.get('CLIP1_PATH', '/workspace/models/CLIP1/clip-vit-large-patch14.safetensors'),
        'CLIP2': os.environ.get('CLIP2_PATH', '/workspace/models/CLIP2/CLIP-ViT-bigG-14-laion2B-39B-b160k.safetensors'),
    }
    
    # Check available RAM
    try:
        import psutil
        available_ram_gb = psutil.virtual_memory().available / (1024**3)
        print(f"Available RAM: {available_ram_gb:.1f} GB", color.CYAN)
        
        # Only preload if we have enough RAM (at least 32GB available)
        if available_ram_gb < 32:
            print(f"Insufficient RAM for preloading models. Skipping.", color.YELLOW)
            return
    except ImportError:
        print("psutil not available, skipping RAM check", color.YELLOW)
        return
    
    print(f"\nPreloading models to CPU memory...", color.MAGENTA)
    start_time = time.time()
    
    # Use ThreadPoolExecutor for parallel loading
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = []
        for name, path in model_paths.items():
            if os.path.exists(path):
                future = executor.submit(preload_model_to_cpu, path, name)
                futures.append(future)
        
        # Wait for all to complete
        for future in as_completed(futures):
            name, state_dict = future.result()
    
    total_time = time.time() - start_time
    print(f"\nAll models preloaded in {total_time:.1f}s", color.MAGENTA)

def get_cached_model(model_name):
    """Get a model from cache if available"""
    with _cache_lock:
        return _model_cache.get(model_name)

def clear_model_cache():
    """Clear the model cache to free memory"""
    global _model_cache
    with _cache_lock:
        _model_cache.clear()
    torch.cuda.empty_cache()
    import gc
    gc.collect()
    print("Model cache cleared", color.YELLOW)

def load_state_dict_fast(model, state_dict_or_path, strict=False):
    """
    Fast loading of state dict, using cache if available
    """
    if isinstance(state_dict_or_path, str):
        # It's a path
        path = state_dict_or_path
        model_name = None
        
        # Try to identify which model this is
        if 'SDXL' in path or 'juggernaut' in path:
            model_name = 'SDXL'
        elif 'SUPIR-v0Q' in path:
            model_name = 'SUPIR_Q'
        elif 'SUPIR-v0F' in path:
            model_name = 'SUPIR_F'
        elif 'CLIP1' in path or 'clip-vit-large-patch14' in path:
            model_name = 'CLIP1'
        elif 'CLIP2' in path or 'ViT-bigG-14' in path:
            model_name = 'CLIP2'
        
        # Check cache first
        if model_name:
            cached = get_cached_model(model_name)
            if cached is not None:
                print(f"Using cached {model_name} from CPU memory", color.GREEN)
                model.load_state_dict(cached, strict=strict)
                return
        
        # Not in cache, load normally
        print(f"Loading from disk: {path}", color.YELLOW)
        if path.endswith('.safetensors'):
            from safetensors.torch import load_file
            state_dict = load_file(path)
        else:
            state_dict = torch.load(path)
        model.load_state_dict(state_dict, strict=strict)
    else:
        # It's already a state dict
        model.load_state_dict(state_dict_or_path, strict=strict)