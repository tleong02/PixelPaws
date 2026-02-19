"""
Find Optimal Batch Size for Your GPU

This script tests different batch sizes to find the fastest one for your system.

Usage:
    python find_optimal_batch_size.py your_video.mp4 your_dlc.h5
"""

import sys
import time
import torch
import cv2
import numpy as np

def test_batch_size(video_file, batch_size):
    """Test processing speed with a specific batch size"""
    print(f"\nTesting batch size: {batch_size}")
    
    cap = cv2.VideoCapture(video_file)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Read some frames
    frames = []
    for _ in range(min(batch_size * 3, 1000)):  # Test with 3 batches
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    
    if len(frames) < batch_size:
        print(f"  Not enough frames, skipping")
        return None
    
    device = torch.device('cuda')
    
    try:
        # Time the batch processing
        start = time.time()
        
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i+batch_size]
            
            # Convert to tensor and move to GPU
            batch_array = np.stack(batch, axis=0)
            batch_tensor = torch.from_numpy(batch_array).to(device).float()
            
            # Grayscale conversion
            gray_batch = (batch_tensor[:, :, :, 0] * 0.299 + 
                         batch_tensor[:, :, :, 1] * 0.587 + 
                         batch_tensor[:, :, :, 2] * 0.114)
            
            # Fake processing (mean of a region)
            result = gray_batch[:, 100:200, 100:200].mean().item()
            
            del batch_tensor, gray_batch
        
        torch.cuda.synchronize()
        elapsed = time.time() - start
        fps = len(frames) / elapsed
        
        print(f"  ✓ Processed {len(frames)} frames in {elapsed:.2f}s ({fps:.0f} fps)")
        return fps
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"  ✗ GPU out of memory!")
            torch.cuda.empty_cache()
            return None
        else:
            raise

def find_optimal_batch_size(video_file):
    """Binary search to find optimal batch size"""
    print("="*60)
    print("Finding Optimal Batch Size")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("✗ No GPU available!")
        return
    
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"Video: {video_file}")
    
    # Test increasing batch sizes
    batch_sizes = [50, 100, 200, 300, 400, 500, 600, 800, 1000]
    results = {}
    
    print("\n" + "="*60)
    print("Testing different batch sizes...")
    print("="*60)
    
    max_working = 50
    best_fps = 0
    best_batch = 50
    
    for batch_size in batch_sizes:
        fps = test_batch_size(video_file, batch_size)
        
        if fps is None:
            print(f"\n✗ Batch size {batch_size} failed (too large)")
            break
        
        results[batch_size] = fps
        max_working = batch_size
        
        if fps > best_fps:
            best_fps = fps
            best_batch = batch_size
        
        # If FPS is decreasing, we've found the sweet spot
        if len(results) > 2:
            last_two = list(results.values())[-2:]
            if last_two[1] < last_two[0] * 0.95:  # 5% slower
                print(f"\n→ FPS decreasing, stopping search")
                break
    
    # Results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"\nBatch Size → Processing Speed:")
    for batch_size, fps in results.items():
        marker = " ← BEST" if batch_size == best_batch else ""
        print(f"  {batch_size:4d} frames → {fps:6.0f} fps{marker}")
    
    print("\n" + "="*60)
    print("RECOMMENDATION")
    print("="*60)
    print(f"\n✓ Use batch size: {best_batch}")
    print(f"  Expected speed: {best_fps:.0f} fps")
    print(f"  Maximum working: {max_working}")
    
    print("\nTo use this batch size, edit brightness_features.py:")
    print(f"  extractor = PixelBrightnessExtractorGPU(..., batch_size={best_batch})")
    
    print("\nOr set in PixelPaws_GUI.py around line 226:")
    print(f"  use_gpu=True, batch_size={best_batch}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python find_optimal_batch_size.py video.mp4")
        print("\nThis will test different batch sizes to find the fastest one")
        sys.exit(1)
    
    video_file = sys.argv[1]
    find_optimal_batch_size(video_file)
