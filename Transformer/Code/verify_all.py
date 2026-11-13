"""
Master Codebase Verification Suite (verify_all.py)

Run this script to automatically verify your Python, PyTorch, CUDA, and Triton setup,
and perform a dry-run execution across all Deep Dive codebase files in the Transformer repository.
"""

import sys
import os
import glob
import importlib.util
import time

def print_header(title):
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)

def check_environment():
    print_header("1. CHECKING ENVIRONMENT & DEPENDENCIES")
    
    # Python Version
    print(f"Python Version: {sys.version.split()[0]} [OK]")
    
    # PyTorch
    try:
        import torch
        print(f"PyTorch Version: {torch.__version__} [OK]")
        cuda_available = torch.cuda.is_available()
        print(f"CUDA Available: {cuda_available}")
        if cuda_available:
            print(f"  GPU Device: {torch.cuda.get_device_name(0)}")
            print(f"  VRAM Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    except ImportError:
        print("CRITICAL: PyTorch is not installed. Run 'pip install torch'.")
        return False

    # NumPy
    try:
        import numpy as np
        print(f"NumPy Version: {np.__version__} [OK]")
    except ImportError:
        print("Warning: NumPy is missing.")

    # Scikit-Learn
    try:
        import sklearn
        print(f"Scikit-Learn Version: {sklearn.__version__} [OK]")
    except ImportError:
        print("Notice: Scikit-Learn is not installed (Required for Classical ML pipelines).")

    # Triton
    try:
        import triton
        print(f"OpenAI Triton Version: {triton.__version__} [OK]")
    except ImportError:
        print("Notice: Triton is not installed (Required for FlashAttention GPU kernels on Linux/WSL).")
        
    return True

def run_deep_dives():
    print_header("2. RUNNING DEEP DIVE CODE VERIFICATION (DRY-RUN)")
    
    deep_dives_dir = os.path.join(os.path.dirname(__file__), "Deep_Dives")
    if not os.path.exists(deep_dives_dir):
        print(f"Error: Could not find directory {deep_dives_dir}")
        return
        
    py_files = sorted(glob.glob(os.path.join(deep_dives_dir, "*.py")))
    print(f"Found {len(py_files)} Deep Dive python scripts to test.\n")
    
    passed = 0
    failed = 0
    skipped = 0
    
    for file_path in py_files:
        file_name = os.path.basename(file_path)
        print(f"Testing {file_name:<45} ... ", end="", flush=True)
        
        start_time = time.time()
        try:
            # Dynamically import the module
            module_name = file_name.replace(".py", "")
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            module = importlib.util.module_from_spec(spec)
            
            # Silence standard stdout during module execution if needed, or let it run
            spec.loader.exec_module(module)
            
            elapsed = time.time() - start_time
            print(f"PASSED ({elapsed:.2f}s)")
            passed += 1
        except SystemExit:
            print("SKIPPED (Clean Exit)")
            skipped += 1
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"FAILED ({elapsed:.2f}s)")
            print(f"   |-- Error: {type(e).__name__}: {e}")
            failed += 1
            
    print("\n" + "-" * 70)
    print(f"VERIFICATION SUMMARY: {passed} Passed | {failed} Failed | {skipped} Skipped | Total: {len(py_files)}")
    print("-" * 70)
    
    if failed == 0:
        print("CONGRATULATIONS! All codebase scripts executed cleanly without errors!")
    else:
        print("Notice: Some files failed. Please check missing dependencies or PyTorch version.")

if __name__ == "__main__":
    if check_environment():
        run_deep_dives()
