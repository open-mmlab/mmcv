"""
Script to replace CUDA and C++ implementations with PyTorch-only implementations.
This script identifies operations that depend on CUDA/C++ and replaces them with
pure PyTorch equivalents.
"""

import os
import re
import glob
from pathlib import Path

# Define paths to check
MMCV_ROOT = Path(__file__).parent
OPS_DIR = MMCV_ROOT / "mmcv" / "ops"

# Pattern to identify ext_loader imports
EXT_LOADER_PATTERN = r"from \.\.utils import ext_loader"
EXT_MODULE_PATTERN = r"ext_module = ext_loader\.load_ext\('_ext',\s*\[([^\]]+)\]\)"

# Functions that have been implemented in pure PyTorch
IMPLEMENTED_FUNCTIONS = {
    'nms': 'from .pure_pytorch_nms import nms_pytorch',
    'softnms': 'from .pure_pytorch_nms import soft_nms_pytorch',
    'nms_match': 'from .pure_pytorch_nms import nms_match_pytorch',
    'nms_quadri': 'from .pure_pytorch_nms import nms_quadri_pytorch',
    'roi_pool_forward': 'from .pure_pytorch_roi import roi_pool_pytorch',
    'roi_pool_backward': None,  # We already handled this
    'roi_align_forward': 'from .pure_pytorch_roi import roi_align_pytorch',
    'roi_align_backward': None,  # We already handled this
}

def process_file(file_path):
    """Process a Python file to replace CUDA/C++ implementations."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Check if this file uses ext_loader
    if not re.search(EXT_LOADER_PATTERN, content):
        return
    
    print(f"Processing {file_path}...")
    
    # Extract the loaded functions
    ext_module_match = re.search(EXT_MODULE_PATTERN, content)
    if not ext_module_match:
        return
    
    functions_str = ext_module_match.group(1)
    functions = [f.strip().strip("'").strip('"') for f in functions_str.split(',')]
    
    # Check if any of these functions have been implemented
    implemented = [f for f in functions if f in IMPLEMENTED_FUNCTIONS]
    if not implemented:
        print(f"  No implementations available for: {functions}")
        return
    
    print(f"  Found {len(implemented)}/{len(functions)} implemented functions")
    
    # Replace the ext_loader import with our implementation imports
    imports = []
    for f in implemented:
        if IMPLEMENTED_FUNCTIONS[f] is not None:
            imports.append(IMPLEMENTED_FUNCTIONS[f])
    
    if imports:
        imports_str = '\n'.join(imports)
        content = re.sub(EXT_LOADER_PATTERN, imports_str, content)
        
        # Remove the ext_module line
        content = re.sub(r"ext_module = ext_loader\.load_ext[^\n]+\n", "", content)
        
        # For each implemented function, we'll need to replace calls to ext_module.function with our implementation
        for f in implemented:
            if f == 'nms':
                content = re.sub(r"ext_module\.nms\(([^)]+)\)", r"nms_pytorch(\1)", content)
            elif f == 'softnms':
                content = re.sub(r"ext_module\.softnms\(([^)]+)\)", r"soft_nms_pytorch(\1)", content)
            elif f == 'nms_match':
                content = re.sub(r"ext_module\.nms_match\(([^)]+)\)", r"nms_match_pytorch(\1)", content)
            elif f == 'nms_quadri':
                content = re.sub(r"ext_module\.nms_quadri\(([^)]+)\)", r"nms_quadri_pytorch(\1)", content)
    
    # Write the modified content back
    with open(file_path, 'w') as f:
        f.write(content)
    print(f"  Updated {file_path}")

def main():
    """Main function to process all Python files in the ops directory."""
    python_files = glob.glob(str(OPS_DIR / "*.py"))
    for py_file in python_files:
        if py_file.endswith("__init__.py") or py_file.endswith("pure_pytorch_nms.py") or py_file.endswith("pure_pytorch_roi.py"):
            continue
        process_file(py_file)
    
    print("\nFinished processing all files.")
    print("\nNOTE: This script does not handle all possible CUDA/C++ operations.")
    print("You will need to manually implement and replace other operations.")
    print("To identify remaining CUDA/C++ dependent operations, look for:")
    print("1. Files that import 'ext_loader'")
    print("2. Usage of 'ext_module' in the code")

if __name__ == "__main__":
    main()