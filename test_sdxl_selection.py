#!/usr/bin/env python3
"""
Test script to verify SDXL model selection functionality
"""

from SUPIR.util import get_available_sdxl_models, create_SUPIR_model
import os

def test_list_models():
    """Test listing available SDXL models"""
    print("Testing get_available_sdxl_models()...")
    models = get_available_sdxl_models()
    
    if not models:
        print("No SDXL models found in models/SDXL/")
    else:
        print(f"Found {len(models)} SDXL models:")
        for model in models:
            print(f"  - {model}")
    
    return models

def test_create_model_with_custom_sdxl():
    """Test creating a SUPIR model with custom SDXL path"""
    print("\nTesting create_SUPIR_model with custom SDXL...")
    
    # Get available models
    models = get_available_sdxl_models()
    if not models:
        print("No models available to test with")
        return
    
    # Use the first available model
    test_model = models[0]
    custom_path = os.path.join("models/SDXL", test_model)
    
    print(f"Creating SUPIR model with custom SDXL: {test_model}")
    
    try:
        # Test with custom SDXL path
        model = create_SUPIR_model("options/SUPIR_v0.yaml", SUPIR_sign="Q", custom_sdxl_path=custom_path)
        print("✓ Successfully created model with custom SDXL path")
        del model  # Clean up
        
        # Test with default (no custom path)
        model = create_SUPIR_model("options/SUPIR_v0.yaml", SUPIR_sign="Q")
        print("✓ Successfully created model with default SDXL path")
        del model  # Clean up
        
    except Exception as e:
        print(f"✗ Error creating model: {e}")

if __name__ == "__main__":
    print("SDXL Model Selection Test")
    print("=" * 50)
    
    # Test 1: List available models
    test_list_models()
    
    # Test 2: Create model with custom SDXL
    test_create_model_with_custom_sdxl()
    
    print("\nTest complete!")