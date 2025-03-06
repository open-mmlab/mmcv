"""
Enhanced script to replace CUDA and C++ implementations with PyTorch-only implementations.
This script identifies operations that depend on CUDA/C++ and replaces them with
pure PyTorch equivalents or stubs with appropriate warnings.
"""

import argparse
import glob
import os
import re
import sys
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
    'roi_pool_backward': None,  # We'll create a dummy implementation
    'roi_align_forward': 'from .pure_pytorch_roi import roi_align_pytorch',
    'roi_align_backward': None,  # We'll create a dummy implementation
    # Add more functions as they are implemented
}

def create_pytorch_fallback_stub(op_name, function_name):
    """Create a stub implementation for a function that doesn't have a PyTorch equivalent yet."""
    stub_content = f"""import torch
import warnings

def {function_name}_pytorch(*args, **kwargs):
    \"\"\"
    PyTorch-only stub implementation of {function_name}.
    This is a placeholder for the C++/CUDA implementation.
    It will raise a warning and return zeros with the appropriate shape.
    
    For production use, a proper PyTorch implementation is needed.
    \"\"\"
    warnings.warn(f"Using stub implementation of {function_name}. "
                 f"This is not a complete implementation and may cause incorrect results.")
    
    # Basic handling depending on the expected output shape
    if len(args) > 0:
        # Try to return something with a reasonable shape
        return torch.zeros_like(args[0])
    
    # Default fallback
    return torch.tensor(0.0)
"""
    
    # Create the directory if it doesn't exist
    pure_pytorch_dir = OPS_DIR / f"pure_pytorch_{op_name}"
    pure_pytorch_dir.mkdir(exist_ok=True)
    
    # Create the file
    stub_file = pure_pytorch_dir / "__init__.py"
    if not stub_file.exists():
        with open(stub_file, 'w') as f:
            f.write("# PyTorch-only implementations for {op_name}\n")
    
    # Write the stub implementation
    impl_file = pure_pytorch_dir / f"{function_name}.py"
    if not impl_file.exists():
        with open(impl_file, 'w') as f:
            f.write(stub_content)
    
    return f"from .pure_pytorch_{op_name}.{function_name} import {function_name}_pytorch"

def process_file(file_path, create_stubs=False):
    """Process a Python file to replace CUDA/C++ implementations."""
    try:
        with open(file_path) as f:
            content = f.read()
    except UnicodeDecodeError:
        print(f"  Skipping {file_path} - not a text file")
        return
    
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
    not_implemented = [f for f in functions if f not in IMPLEMENTED_FUNCTIONS]
    
    # Create stubs for functions that haven't been implemented
    additional_imports = []
    if create_stubs and not_implemented:
        print(f"  Creating stubs for {len(not_implemented)} functions")
        for func in not_implemented:
            # Extract the base op name from the filename
            base_name = os.path.basename(file_path).replace('.py', '')
            import_line = create_pytorch_fallback_stub(base_name, func)
            additional_imports.append(import_line)
            IMPLEMENTED_FUNCTIONS[func] = import_line
            implemented.append(func)
    
    if not implemented:
        print(f"  No implementations available for: {functions}")
        return
    
    print(f"  Found {len(implemented)}/{len(functions)} implemented or stubbed functions")
    
    # Replace the ext_loader import with our implementation imports
    imports = []
    for f in implemented:
        if IMPLEMENTED_FUNCTIONS[f] is not None:
            imports.append(IMPLEMENTED_FUNCTIONS[f])
    
    imports.extend(additional_imports)
    
    if imports:
        imports_str = '\n'.join(imports)
        content = re.sub(EXT_LOADER_PATTERN, imports_str, content)
        
        # Remove the ext_module line
        content = re.sub(r"ext_module = ext_loader\.load_ext[^\n]+\n", "", content)
        
        # Replace function calls with their PyTorch equivalents
        for func in implemented:
            # Generic replacement pattern
            pattern = rf"ext_module\.{func}\(([^)]+)\)"
            replacement = f"{func}_pytorch(\\1)"
            content = re.sub(pattern, replacement, content)
    
    # Write the modified content back
    with open(file_path, 'w') as f:
        f.write(content)
    print(f"  Updated {file_path}")

def list_unimplemented_ops():
    """List all operations that still use ext_loader."""
    unimplemented = []
    python_files = glob.glob(str(OPS_DIR / "*.py"))
    
    for py_file in python_files:
        if py_file.endswith("__init__.py") or py_file.endswith("pure_pytorch_"):
            continue
            
        try:
            with open(py_file) as f:
                content = f.read()
        except UnicodeDecodeError:
            continue
            
        if re.search(EXT_LOADER_PATTERN, content):
            ext_module_match = re.search(EXT_MODULE_PATTERN, content)
            if ext_module_match:
                functions_str = ext_module_match.group(1)
                functions = [f.strip().strip("'").strip('"') for f in functions_str.split(',')]
                unimplemented.append((os.path.basename(py_file), functions))
    
    return unimplemented

def main():
    """Main function to process all Python files in the ops directory."""
    parser = argparse.ArgumentParser(description='Replace CUDA/C++ implementations with PyTorch equivalents')
    parser.add_argument('--create-stubs', action='store_true', help='Create stub implementations for unimplemented functions')
    parser.add_argument('--list-unimplemented', action='store_true', help='List all operations still using ext_loader')
    parser.add_argument('--file', type=str, help='Process a specific file only')
    
    args = parser.parse_args()
    
    if args.list_unimplemented:
        print("Operations still using ext_loader:")
        for op, funcs in list_unimplemented_ops():
            print(f"  {op}: {', '.join(funcs)}")
        sys.exit(0)
    
    if args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"Error: File {args.file} not found")
            sys.exit(1)
        process_file(file_path, args.create_stubs)
    else:
        python_files = glob.glob(str(OPS_DIR / "*.py"))
        for py_file in python_files:
            if py_file.endswith("__init__.py") or "pure_pytorch_" in py_file:
                continue
            process_file(py_file, args.create_stubs)
    
    print("\nFinished processing files.")
    
    remaining = list_unimplemented_ops()
    if remaining:
        print(f"\nThere are still {len(remaining)} operations using ext_loader:")
        for op, funcs in remaining:
            print(f"  {op}: {', '.join(funcs)}")
        print("\nTo create stub implementations for all of them, run:")
        print("  python remove_cuda_deps.py --create-stubs")
    else:
        print("\nAll operations have been converted to use PyTorch-only implementations!")

if __name__ == "__main__":
    main()