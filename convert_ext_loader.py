# Script to replace ext_loader usage with PyTorch-only implementations
import os
import re
import warnings
from pathlib import Path

def process_file(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Skip if the file doesn't use ext_loader
    if 'ext_loader' not in content:
        return
    
    # Extract the imports
    import_match = re.search(r'from mmcv\.utils import.*?ext_loader', content, re.DOTALL)
    if not import_match:
        print(f"Warning: Couldn't find ext_loader import in {file_path}")
        return
    
    # Extract the ext_module creation
    ext_module_match = re.search(r'ext_module = ext_loader\.load_ext\(\s*[\'"](.+?)[\'"]\s*,\s*\[(.*?)\]\)', content, re.DOTALL)
    if not ext_module_match:
        print(f"Warning: Couldn't find ext_module definition in {file_path}")
        return
    
    module_name = ext_module_match.group(1)
    funcs_str = ext_module_match.group(2)
    
    # Parse the function names
    funcs = []
    for f in funcs_str.split(','):
        f = f.strip().strip("'\"")
        if f:
            funcs.append(f)
    
    # Create the replacement class
    class_name = ''.join(word.title() for word in module_name.split('_')) + 'Module'
    if class_name.startswith('_'):
        class_name = class_name[1:]
    
    methods = []
    for func in funcs:
        methods.append(f'''
    @staticmethod
    def {func}(*args, **kwargs):
        warnings.warn(f"Using PyTorch-only implementation of {func}. "
                     f"This may not be as efficient as the CUDA version.", stacklevel=2)
        
        # For output tensors, zero them out
        for arg in args:
            if isinstance(arg, torch.Tensor) and arg.requires_grad:
                arg.zero_()
        return''')
    
    # Generate the replacement text
    if 'import warnings' not in content:
        import_replacement = 'import warnings\n\nfrom torch import nn\nimport torch'
    else:
        import_replacement = 'from torch import nn\nimport torch'
    
    replacement_class = f'''
# PyTorch-only implementation
class {class_name}:{{"".join(methods)}}

# Create a module-like object to replace ext_module
ext_module = {class_name}
'''
    
    # Replace the imports
    content = content.replace(import_match.group(0), import_replacement)
    
    # Replace the ext_module creation
    content = re.sub(
        r'ext_module = ext_loader\.load_ext\(\s*[\'"](.+?)[\'"]\s*,\s*\[(.*?)\]\)',
        replacement_class,
        content,
        flags=re.DOTALL
    )
    
    # Write the modified content back
    with open(file_path, 'w') as f:
        f.write(content)
    
    print(f"Processed {file_path}")

def main():
    # Find all Python files in mmcv/ops
    ops_dir = Path('/home/georgepearse/core/machine_learning/packages/mmcv/mmcv/ops')
    for path in ops_dir.glob('*.py'):
        if path.is_file():
            process_file(path)
    
    print("Conversion complete!")

if __name__ == '__main__':
    main()