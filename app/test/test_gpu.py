#!/usr/bin/env python3
"""
GPU Test Script for GLASS Inference
Tests GPU availability, performance, and compatibility
"""

import torch
import cv2
import numpy as np
import time
import sys
import os
from pathlib import Path

def test_pytorch_gpu():
    """Test PyTorch GPU availability and basic operations"""
    print("=" * 60)
    print("🔍 PYTORCH GPU TEST")
    print("=" * 60)
    
    # Basic CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    
    if not cuda_available:
        print("❌ CUDA not available - GPU inference will not work")
        return False
    
    # GPU count and details
    gpu_count = torch.cuda.device_count()
    print(f"GPU Count: {gpu_count}")
    
    for i in range(gpu_count):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_props = torch.cuda.get_device_properties(i)
        gpu_memory = gpu_props.total_memory / (1024**3)
        compute_capability = f"{gpu_props.major}.{gpu_props.minor}"
        print(f"  GPU {i}: {gpu_name}")
        print(f"    Memory: {gpu_memory:.1f} GB")
        print(f"    Compute Capability: {compute_capability}")
        print(f"    Multiprocessors: {gpu_props.multi_processor_count}")
    
    # PyTorch versions
    print(f"\nPyTorch Version: {torch.__version__}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"cuDNN Available: {torch.backends.cudnn.is_available()}")
    if torch.backends.cudnn.is_available():
        print(f"cuDNN Version: {torch.backends.cudnn.version()}")
    
    return True

def test_gpu_performance():
    """Test GPU vs CPU performance with tensor operations"""
    print("\n" + "=" * 60)
    print("⚡ GPU PERFORMANCE TEST")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("❌ Skipping performance test - CUDA not available")
        return
    
    # Test tensor operations
    device_cpu = torch.device('cpu')
    device_gpu = torch.device('cuda:0')
    
    # Create test tensors (simulating image batch)
    batch_size = 8
    channels = 3
    height = 384
    width = 384
    
    print(f"Testing with tensor size: [{batch_size}, {channels}, {height}, {width}]")
    
    # CPU test
    print("\n🐌 CPU Performance:")
    tensor_cpu = torch.randn(batch_size, channels, height, width, device=device_cpu)
    
    start_time = time.time()
    for _ in range(10):
        result_cpu = torch.nn.functional.conv2d(tensor_cpu, torch.randn(64, channels, 3, 3, device=device_cpu))
    cpu_time = time.time() - start_time
    print(f"  10 convolutions: {cpu_time*1000:.1f}ms")
    print(f"  Average per operation: {cpu_time*100:.1f}ms")
    
    # GPU test
    print("\n🚀 GPU Performance:")
    tensor_gpu = torch.randn(batch_size, channels, height, width, device=device_gpu)
    
    # Warmup GPU
    for _ in range(5):
        _ = torch.nn.functional.conv2d(tensor_gpu, torch.randn(64, channels, 3, 3, device=device_gpu))
    torch.cuda.synchronize()
    
    start_time = time.time()
    for _ in range(10):
        result_gpu = torch.nn.functional.conv2d(tensor_gpu, torch.randn(64, channels, 3, 3, device=device_gpu))
    torch.cuda.synchronize()
    gpu_time = time.time() - start_time
    print(f"  10 convolutions: {gpu_time*1000:.1f}ms")
    print(f"  Average per operation: {gpu_time*100:.1f}ms")
    
    # Performance comparison
    speedup = cpu_time / gpu_time if gpu_time > 0 else 0
    print(f"\n📊 Performance Summary:")
    print(f"  GPU Speedup: {speedup:.1f}x faster than CPU")
    
    if speedup > 5:
        print("  ✅ Excellent GPU performance!")
    elif speedup > 2:
        print("  ✅ Good GPU performance")
    else:
        print("  ⚠️  GPU performance lower than expected")

def test_gpu_memory():
    """Test GPU memory allocation and usage"""
    print("\n" + "=" * 60)
    print("🧠 GPU MEMORY TEST")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("❌ Skipping memory test - CUDA not available")
        return
    
    device = torch.device('cuda:0')
    
    # Initial memory state
    torch.cuda.empty_cache()
    initial_allocated = torch.cuda.memory_allocated(device) / (1024**3)
    initial_reserved = torch.cuda.memory_reserved(device) / (1024**3)
    total_memory = torch.cuda.get_device_properties(device).total_memory / (1024**3)
    
    print(f"Total GPU Memory: {total_memory:.1f} GB")
    print(f"Initial Allocated: {initial_allocated:.3f} GB")
    print(f"Initial Reserved: {initial_reserved:.3f} GB")
    
    # Allocate test tensors
    print("\n📈 Allocating test tensors...")
    tensors = []
    
    try:
        for i in range(5):
            # Allocate ~500MB tensor
            tensor = torch.randn(1, 3, 2048, 2048, device=device)
            tensors.append(tensor)
            
            allocated = torch.cuda.memory_allocated(device) / (1024**3)
            reserved = torch.cuda.memory_reserved(device) / (1024**3)
            print(f"  Step {i+1}: Allocated: {allocated:.3f} GB, Reserved: {reserved:.3f} GB")
            
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"  ❌ Out of memory at step {len(tensors)+1}")
            print(f"  💡 Available memory: ~{len(tensors) * 0.5:.1f} GB for inference")
        else:
            print(f"  ❌ Error: {e}")
    
    # Clean up
    del tensors
    torch.cuda.empty_cache()
    
    final_allocated = torch.cuda.memory_allocated(device) / (1024**3)
    final_reserved = torch.cuda.memory_reserved(device) / (1024**3)
    print(f"\n🧹 After cleanup:")
    print(f"  Allocated: {final_allocated:.3f} GB")
    print(f"  Reserved: {final_reserved:.3f} GB")

def test_opencv_gpu():
    """Test OpenCV GPU support"""
    print("\n" + "=" * 60)
    print("📹 OPENCV GPU TEST")
    print("=" * 60)
    
    # Check OpenCV CUDA support
    cuda_devices = cv2.cuda.getCudaEnabledDeviceCount()
    print(f"OpenCV CUDA Devices: {cuda_devices}")
    
    if cuda_devices > 0:
        print("✅ OpenCV compiled with CUDA support")
        
        # Test GPU mat operations
        try:
            # Create test image
            img_cpu = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
            
            # Upload to GPU
            gpu_mat = cv2.cuda_GpuMat()
            gpu_mat.upload(img_cpu)
            
            # Simple GPU operation
            gpu_gray = cv2.cuda.cvtColor(gpu_mat, cv2.COLOR_BGR2GRAY)
            
            # Download result
            result = gpu_gray.download()
            
            print("✅ OpenCV GPU operations working")
            print(f"  Processed image: {img_cpu.shape} -> {result.shape}")
            
        except Exception as e:
            print(f"❌ OpenCV GPU operations failed: {e}")
    else:
        print("❌ OpenCV not compiled with CUDA support")
        print("💡 Camera operations will use CPU only")

def test_glass_compatibility():
    """Test GLASS model compatibility"""
    print("\n" + "=" * 60)
    print("🔬 GLASS COMPATIBILITY TEST")
    print("=" * 60)
    
    try:
        # Add GLASS paths
        glass_root = Path(__file__).parent
        inference_path = glass_root / "inference"
        
        if inference_path.exists():
            sys.path.insert(0, str(inference_path))
            
            # Test GLASS imports
            try:
                import backbones
                import glass
                print("✅ GLASS modules imported successfully")
                
                # Test model path
                models_path = glass_root / "results" / "models" / "backbone_0"
                if models_path.exists():
                    model_dirs = [d for d in models_path.iterdir() if d.is_dir()]
                    print(f"✅ Found {len(model_dirs)} model directories")
                    for model_dir in model_dirs:
                        print(f"  - {model_dir.name}")
                else:
                    print("⚠️  No trained models found")
                    print(f"  Expected path: {models_path}")
                
            except ImportError as e:
                print(f"❌ GLASS import failed: {e}")
                
        else:
            print("❌ GLASS inference directory not found")
            print(f"  Expected: {inference_path}")
            
    except Exception as e:
        print(f"❌ GLASS compatibility test failed: {e}")

def main():
    """Run all GPU tests"""
    print("🧪 GLASS GPU COMPATIBILITY TEST")
    print("Testing GPU setup for fabric inspection inference")
    print()
    
    # Run all tests
    gpu_available = test_pytorch_gpu()
    
    if gpu_available:
        test_gpu_performance()
        test_gpu_memory()
    
    test_opencv_gpu()
    test_glass_compatibility()
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 TEST SUMMARY")
    print("=" * 60)
    
    if gpu_available:
        print("✅ PyTorch GPU support: AVAILABLE")
        print("🚀 Recommendation: GPU inference will work")
        print("💡 Expected performance: 5-10x faster than CPU")
    else:
        print("❌ PyTorch GPU support: NOT AVAILABLE")
        print("🐌 Recommendation: Will fallback to CPU inference")
        print("💡 Consider installing CUDA and PyTorch with GPU support")
    
    print("\n🔧 To install GPU support:")
    print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    print("\n📖 For more info, visit: https://pytorch.org/get-started/locally/")

if __name__ == "__main__":
    main()
