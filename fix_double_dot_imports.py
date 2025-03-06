#!/usr/bin/env python3

import os
import re

def fix_double_dot_imports(directory):
    """Find and fix instances of 'from mmcv.ops..' to 'from mmcv.ops.'"""
    pattern = r'from mmcv\.ops\.\.'
    replacement = r'from mmcv.ops.'
    
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                
                # Read the file
                with open(file_path, 'r') as f:
                    content = f.read()
                
                # Check if the pattern exists
                if re.search(pattern, content):
                    # Replace the pattern
                    new_content = re.sub(pattern, replacement, content)
                    
                    # Write the changes back
                    with open(file_path, 'w') as f:
                        f.write(new_content)
                    
                    print(f"Fixed: {file_path}")

if __name__ == "__main__":
    # Fix in the mmcv directory
    fix_double_dot_imports("mmcv")
    print("Done!")