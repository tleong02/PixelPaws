"""
Enable PyTorch GPU Acceleration for PixelPaws

This script swaps the brightness feature extractor to use PyTorch GPU acceleration.

Usage:
    1. Install PyTorch with CUDA:
       pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    
    2. Run this script:
       python enable_pytorch_gpu.py
    
    3. Launch PixelPaws normally - it will now use GPU!

To revert:
    python enable_pytorch_gpu.py --revert
"""

import os
import shutil
import argparse

def enable_pytorch_gpu():
    """Replace brightness_features.py with PyTorch GPU version"""
    
    # Check if files exist
    original = "brightness_features.py"
    pytorch_version = "brightness_features_pytorch.py"
    backup = "brightness_features_cpu_backup.py"
    
    if not os.path.exists(pytorch_version):
        print(f"✗ Error: {pytorch_version} not found!")
        print("  Make sure both files are in the same directory.")
        return False
    
    if not os.path.exists(original):
        print(f"✗ Error: {original} not found!")
        return False
    
    # Create backup of original
    if not os.path.exists(backup):
        print(f"Creating backup: {backup}")
        shutil.copy2(original, backup)
    
    # Replace with PyTorch version
    print(f"Enabling PyTorch GPU mode...")
    shutil.copy2(pytorch_version, original)
    
    print("✓ PyTorch GPU mode enabled!")
    print("\nNext steps:")
    print("  1. Make sure PyTorch with CUDA is installed:")
    print("     pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
    print("  2. Launch PixelPaws normally")
    print("  3. Feature extraction will now use GPU!")
    print("\nTo revert: python enable_pytorch_gpu.py --revert")
    
    return True

def revert_to_cpu():
    """Restore original CPU version"""
    
    original = "brightness_features.py"
    backup = "brightness_features_cpu_backup.py"
    
    if not os.path.exists(backup):
        print(f"✗ Error: Backup file {backup} not found!")
        print("  Cannot revert to original version.")
        return False
    
    print(f"Reverting to CPU version...")
    shutil.copy2(backup, original)
    
    print("✓ Reverted to CPU version!")
    print("\nCPU mode restored. Feature extraction will use OpenCV (CPU).")
    
    return True

def check_pytorch_cuda():
    """Check if PyTorch with CUDA is installed"""
    try:
        import torch
        print("\n" + "="*60)
        print("PyTorch Status:")
        print("="*60)
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            print("\n✓ PyTorch with CUDA is ready!")
        else:
            print("\n⚠ PyTorch found but CUDA not available")
            print("\nTo install PyTorch with CUDA:")
            print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
        
        return torch.cuda.is_available()
        
    except ImportError:
        print("\n✗ PyTorch not installed")
        print("\nTo install PyTorch with CUDA:")
        print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enable/disable PyTorch GPU acceleration")
    parser.add_argument('--revert', action='store_true', 
                       help="Revert to CPU version")
    parser.add_argument('--check', action='store_true',
                       help="Check PyTorch CUDA status")
    
    args = parser.parse_args()
    
    if args.check:
        check_pytorch_cuda()
    elif args.revert:
        revert_to_cpu()
    else:
        # Check PyTorch first
        has_cuda = check_pytorch_cuda()
        
        if not has_cuda:
            print("\n" + "="*60)
            response = input("PyTorch CUDA not available. Continue anyway? (y/n): ")
            if response.lower() != 'y':
                print("Aborted.")
                exit(0)
        
        print("\n" + "="*60)
        enable_pytorch_gpu()
