import torch
import torch.nn as nn
import time
import copy

from swin4d_transformer_ver7 import SwinTransformer4D as OriginalSwin
from swin4d_transformer_fix import SwinTransformer4D as FixedSwin

def run_benchmark(model, x, label="Model", num_iters=20):
    if model is None:
        print(f"Skipping {label} as it was not imported.")
        return

    # Move to GPU and set to eval mode
    model = model.cuda().eval()
    
    with torch.no_grad():
        # 1. Warm-up Pass (Crucial for Fixed model to cache the mask)
        _ = model(x)
        torch.cuda.synchronize()
        
        # 2. Timing Loop
        start_time = time.time()
        for _ in range(num_iters):
            _ = model(x)
        torch.cuda.synchronize()
        end_time = time.time()
        
        avg_latency = ((end_time - start_time) / num_iters) * 1000 # Convert to ms
        
        # 3. Memory Stats
        peak_vram = torch.cuda.max_memory_allocated() / (1024**2) # MB
        
        print(f"{label:10} | Avg Latency: {avg_latency:8.2f} ms | Peak VRAM: {peak_vram:8.2f} MB")
        
        # Reset stats for next run
        torch.cuda.reset_peak_memory_stats()

def main():
    if not torch.cuda.is_available():
        print("CUDA is not available. This benchmark requires a GPU.")
        return

    # Input Shape: (Batch, Channels, Depth, Height, Width, Time)
    input_shape = (1, 1, 64, 64, 64, 10)
    dummy_input = torch.randn(input_shape).cuda()
    
    # Standard SwinTransformer4D parameters
    params = {
        "img_size": (64, 64, 64, 10),
        "in_chans": 1,
        "embed_dim": 48,
        "window_size": (4, 4, 4, 4),
        "first_window_size": (4, 4, 4, 4),
        "patch_size": (2, 2, 2, 1), # Temporal patch size MUST be 1
        "depths": (2, 2, 6, 2),
        "num_heads": (3, 6, 12, 24),
        "spatial_dims": 4,
    }

    print(f"--- Swin4D Optimization Benchmark ---")
    print(f"Input Shape: {input_shape}")
    print("-" * 60)

    # 1. Initialize Original Model
    m_orig = OriginalSwin(**params).cuda().eval()
    
    # 2. Initialize Fixed Model and SYNC WEIGHTS
    m_fix = FixedSwin(**params).cuda().eval()
    # load_state_dict ensures both models start with the exact same numerical values
    m_fix.load_state_dict(m_orig.state_dict(), strict=False)

    # Run Benchmark for Original
    run_benchmark(m_orig, dummy_input, "Original")
    
    # Clean up Original model from VRAM completely before benchmarking Fixed
    del m_orig
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # Run Benchmark for Fixed
    run_benchmark(m_fix, dummy_input, "Fixed")

    # Verification of Numerical Correctness
    print("-" * 60)
    print("Verifying numerical consistency...")
    with torch.no_grad():
        out_orig_check = FixedSwin(**params).cuda().eval() # Using another instance to be safe
        out_orig_check.load_state_dict(m_fix.state_dict(), strict=False)
        
        # We compare the output of Fixed against its own weights logic to prove no drift
        # But to be absolutely sure, we use the m_fix instance we just benchmarked
        out_fix = m_fix(dummy_input)
        
        # We need a reference from a fresh OriginalSwin with same weights
        m_orig_ref = OriginalSwin(**params).cuda().eval()
        m_orig_ref.load_state_dict(m_fix.state_dict(), strict=False)
        out_orig = m_orig_ref(dummy_input)
        
        # Check maximum absolute difference
        max_diff = torch.abs(out_orig - out_fix).max().item()
        print(f"Max numerical difference: {max_diff:.2e}")
        if max_diff < 1e-4:
            print("SUCCESS: Models are numerically identical (within float32 precision).")
        else:
            print("WARNING: Numerical difference detected. Check weight syncing.")

if __name__ == "__main__":
    main()
