"""
RunPod Serverless Handler for SUPIR Image Enhancement
"""
import runpod
import torch
import torch.cuda
import base64
import io
import os
import time
from PIL import Image
from typing import Dict, Any, List, Optional

from SUPIR.util import create_SUPIR_model, PIL2Tensor, Tensor2PIL, convert_dtype

# Global model variable to persist between requests
model = None
device = None

def load_model():
    """Load the SUPIR model into memory"""
    global model, device
    
    if model is not None:
        return model
    
    print("Loading SUPIR model...")
    
    # Check for CUDA availability
    if torch.cuda.is_available():
        device = 'cuda:0'
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print("CUDA not available. Using CPU.")
    
    # Default configuration
    config = "options/SUPIR_v0_tiled.yaml"  # Use tiled for memory efficiency
    
    # Create model
    model = create_SUPIR_model(config, SUPIR_sign='Q')  # Default to Q mode
    model = model.half()  # Use half precision for memory efficiency
    model.init_tile_vae(encoder_tile_size=512, decoder_tile_size=64)
    model.ae_dtype = convert_dtype("bf16")
    model.model.dtype = convert_dtype("fp16")
    model = model.to(device)
    
    # Set tile parameters for TiledRestoreEDMSampler
    model.sampler.tile_size = 128
    model.sampler.tile_stride = 64
    
    print("Model loaded successfully")
    return model

def process_image(input_image: Image.Image, params: Dict[str, Any]) -> Image.Image:
    """Process a single image with SUPIR"""
    global model, device
    
    # Extract parameters with defaults
    upscale = params.get('upscale', 2)
    SUPIR_sign = params.get('SUPIR_sign', 'Q')
    img_caption = params.get('img_caption', '')
    edm_steps = params.get('edm_steps', 50)
    restoration_scale = params.get('restoration_scale', -1)
    cfg_scale_start = params.get('cfg_scale_start', 2.0)
    cfg_scale_end = params.get('cfg_scale_end', 4.0)
    control_scale_start = params.get('control_scale_start', 0.9)
    control_scale_end = params.get('control_scale_end', 0.9)
    color_fix_type = params.get('color_fix_type', 'Wavelet')
    seed = params.get('seed', 1234567891)
    
    # Additional prompts
    a_prompt = params.get('a_prompt', 
        'Cinematic, High Contrast, highly detailed, taken using a Canon EOS R '
        'camera, hyper detailed photo - realistic maximum detail, 32k, Color '
        'Grading, ultra HD, extreme meticulous detailing, skin pore detailing, '
        'hyper sharpness, perfect without deformations.')
    n_prompt = params.get('n_prompt',
        'painting, oil painting, illustration, drawing, art, sketch, oil painting, '
        'cartoon, CG Style, 3D render, unreal engine, blurring, dirty, messy, '
        'worst quality, low quality, frames, watermark, signature, jpeg artifacts, '
        'deformed, lowres, over-smooth')
    
    # Convert image to tensor
    LQ_img, h0, w0 = PIL2Tensor(input_image, upscale=upscale, min_size=1024)
    LQ_img = LQ_img.unsqueeze(0).to(device)[:, :3, :, :]
    
    # Process with SUPIR
    samples = model.batchify_sample(
        LQ_img, img_caption,
        num_steps=edm_steps,
        restoration_scale=restoration_scale,
        s_churn=5,
        s_noise=1.003,
        cfg_scale_start=cfg_scale_start,
        cfg_scale_end=cfg_scale_end,
        control_scale_start=control_scale_start,
        control_scale_end=control_scale_end,
        seed=seed,
        num_samples=1,
        p_p=a_prompt,
        n_p=n_prompt,
        color_fix_type=color_fix_type,
        skip_denoise_stage=False
    )
    
    # Convert back to PIL Image
    output_image = Tensor2PIL(samples[0], h0, w0)
    return output_image

def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod serverless handler function
    
    Expected input format:
    {
        "input": {
            "image": "base64_encoded_image_string",
            "params": {
                "upscale": 2,
                "SUPIR_sign": "Q",
                "img_caption": "optional caption",
                ... other optional parameters
            }
        }
    }
    
    Returns:
    {
        "output": {
            "image": "base64_encoded_output_image",
            "processing_time": float,
            "parameters_used": dict
        }
    }
    """
    try:
        start_time = time.time()
        
        # Load model if not already loaded
        load_model()
        
        # Extract input
        job_input = job.get("input", {})
        
        # Decode base64 image
        image_data = job_input.get("image")
        if not image_data:
            return {"error": "No image provided in input"}
        
        # Handle base64 prefix if present
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        
        # Decode and open image
        image_bytes = base64.b64decode(image_data)
        input_image = Image.open(io.BytesIO(image_bytes))
        
        # Get processing parameters
        params = job_input.get("params", {})
        
        # Process the image
        output_image = process_image(input_image, params)
        
        # Convert output to base64
        output_buffer = io.BytesIO()
        output_image.save(output_buffer, format='PNG')
        output_buffer.seek(0)
        output_base64 = base64.b64encode(output_buffer.getvalue()).decode('utf-8')
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        return {
            "output": {
                "image": output_base64,
                "processing_time": processing_time,
                "parameters_used": params,
                "output_size": f"{output_image.width}x{output_image.height}"
            }
        }
        
    except Exception as e:
        print(f"Error in handler: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }

# RunPod serverless entry point
runpod.serverless.start({"handler": handler})