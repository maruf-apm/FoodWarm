#!/usr/bin/env python3
"""GPU Diagnostic for Colab."""

import torch
import subprocess

print("=" * 60)
print("GPU DIAGNOSTIC")
print("=" * 60)

result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
print(result.stdout)

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {vram:.1f} GB")

    x = torch.randn(5000, 5000, device="cuda")
    y = x @ x.T
    print(f"✅ Test tensor: {y.shape} on {y.device}")
    del x, y
    torch.cuda.empty_cache()
    print("\n✅ GPU IS READY")
else:
    print("\n❌ NO GPU! Runtime → Change runtime type → GPU → Restart")
