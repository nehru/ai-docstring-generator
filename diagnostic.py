#!/usr/bin/env python3
"""
Diagnostic script to check system setup for fine-tuning
"""

import sys
print("="*60)
print("SYSTEM DIAGNOSTIC CHECK")
print("="*60)

# Check Python version
print(f"\n1. Python Version: {sys.version}")

# Check if PyTorch is installed
print("\n2. Checking PyTorch...")
try:
    import torch
    print(f"   PyTorch installed: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("   WARNING: No GPU detected - training will be very slow")
except ImportError as e:
    print(f"   ERROR: PyTorch not installed: {e}")
    print("   Install: pip install torch --index-url https://download.pytorch.org/whl/cu118")
    sys.exit(1)

# Check transformers
print("\n3. Checking Transformers...")
try:
    import transformers
    print(f"   Transformers installed: {transformers.__version__}")
except ImportError as e:
    print(f"   ERROR: Transformers not installed: {e}")
    print("   Install: pip install transformers")
    sys.exit(1)

# Check datasets
print("\n4. Checking Datasets...")
try:
    import datasets
    print(f"   Datasets installed: {datasets.__version__}")
except ImportError as e:
    print(f"   ERROR: Datasets not installed: {e}")
    print("   Install: pip install datasets")
    sys.exit(1)

# Check PEFT
print("\n5. Checking PEFT...")
try:
    import peft
    print(f"   PEFT installed: {peft.__version__}")
except ImportError as e:
    print(f"   ERROR: PEFT not installed: {e}")
    print("   Install: pip install peft")
    sys.exit(1)

# Check accelerate
print("\n6. Checking Accelerate...")
try:
    import accelerate
    print(f"   Accelerate installed: {accelerate.__version__}")
except ImportError as e:
    print(f"   ERROR: Accelerate not installed: {e}")
    print("   Install: pip install accelerate")
    sys.exit(1)

# Check if dataset.json exists
print("\n7. Checking dataset file...")
import os
if os.path.exists("dataset.json"):
    print("   dataset.json found")
    import json
    try:
        with open("dataset.json", 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"   Dataset size: {len(data)} examples")
        print("   Dataset is valid JSON")
    except json.JSONDecodeError as e:
        print(f"   ERROR: dataset.json is corrupted: {e}")
        print("   Run: python fix_dataset.py")
        sys.exit(1)
else:
    print("   ERROR: dataset.json not found")
    print("   Make sure dataset.json is in the same directory")
    sys.exit(1)

# Test basic model access
print("\n8. Testing model access...")
print("   (This may download model files on first run)")
try:
    from transformers import AutoTokenizer
    print("   Attempting to load tokenizer...")
    
    test_model = "microsoft/phi-3-mini-4k-instruct"
    print(f"   Testing: {test_model}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        test_model,
        trust_remote_code=True
    )
    print("   Model access successful")
    print("   Note: Full model will download when running fine_tune_phi3_final.py")
    
except Exception as e:
    print(f"   WARNING: Model test failed: {e}")
    print("   This is normal if you don't have internet connection")
    print("   The script will attempt download when you run fine_tune_phi3_final.py")

print("\n" + "="*60)
print("DIAGNOSTIC COMPLETE")
print("="*60)
print("\nNext steps:")
print("1. If all checks passed, run: python fine_tune_phi3_final.py")
print("2. If PyTorch/CUDA issues, reinstall: pip install torch --force-reinstall")
print("3. If missing packages, run: pip install -r requirements.txt")
print("\nNote: First run will download model files (~7 GB)")
print("This may take 10-15 minutes depending on internet speed")