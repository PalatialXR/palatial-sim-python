import torch
import psutil
import os
import gc

def print_memory_stats():
    """Print current GPU and CPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        free = total - allocated
        
        print("\nGPU Memory Stats:")
        print(f"Device: {torch.cuda.get_device_name(0)}")
        print(f"Allocated: {allocated:.2f} GB")
        print(f"Reserved: {reserved:.2f} GB")
        print(f"Total: {total:.2f} GB")
        print(f"Free: {free:.2f} GB")
        print(f"Utilization: {(allocated/total)*100:.1f}%")

    print("\nCPU Memory Stats:")
    print(f"Process Memory: {psutil.Process().memory_info().rss / (1024**3):.2f} GB")

def clear_gpu_memory(aggressive=False):
    """Clear GPU memory cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if aggressive:
            gc.collect()
            torch.cuda.reset_peak_memory_stats()

def optimize_memory_usage(enable=True):
    """Configure PyTorch for optimized memory usage"""
    if enable:
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True