#!/usr/bin/env python3
"""Google Colab setup script for Foodwarm.

Run this cell first in your Colab notebook:

    !wget https://raw.githubusercontent.com/YOUR_USERNAME/foodwarm/main/scripts/colab_setup.py
    !python colab_setup.py

Or manually:
    !git clone https://github.com/YOUR_USERNAME/foodwarm.git
    %cd foodwarm
    !pip install -r requirements.txt
"""
import os
import subprocess


def setup_colab():
    print("=" * 60)
    print("FOODWARM - Google Colab Setup")
    print("=" * 60)

    # Check GPU
    result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], 
                          capture_output=True, text=True)
    if result.returncode == 0:
        gpu_info = result.stdout.strip()
        print(f"✅ GPU detected: {gpu_info}")
    else:
        print("⚠️  No GPU detected! Make sure you selected GPU runtime.")
        print("   Runtime → Change runtime type → Hardware accelerator: GPU")
        return

    # Install dependencies
    print("\n📦 Installing dependencies...")
    subprocess.run(["pip", "install", "-q", "-r", "requirements.txt"])
    print("✅ Dependencies installed")

    # Create data directory
    os.makedirs("./data", exist_ok=True)
    os.makedirs("./outputs", exist_ok=True)

    # Check available VRAM
    import torch
    if torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"\n💾 Available VRAM: {vram:.1f} GB")

        if vram < 12:
            print("   ⚠️  Low VRAM detected. Use configs/colab/linear_colab.yaml")
        elif vram < 16:
            print("   ✅ Good for LoRA. Use configs/colab/lora_colab.yaml")
        else:
            print("   ✅ Excellent! You can even try full fine-tuning.")

    print("\n" + "=" * 60)
    print("Setup complete! Ready to train.")
    print("=" * 60)
    print("\nQuick test:")
    print("  !python scripts/test_train.py --config configs/colab/lora_colab.yaml")
    print("\nFull training:")
    print("  !python scripts/train.py --config configs/colab/lora_colab.yaml")


if __name__ == "__main__":
    setup_colab()
