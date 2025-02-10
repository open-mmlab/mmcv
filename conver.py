import os
import re

def replace_in_file(file_path, old_str, new_str):
    """在文件中替换字符串，同时保持大小写一致性"""
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    def replacer(match):
        word = match.group(0)
        if word.lower() == old_str.lower():
            # 计算新的单词并保持大小写一致
            new_word = ''.join(new_str[i].upper() if c.isupper() else new_str[i].lower()
                               for i, c in enumerate(word))
            return new_word
        return word

    pattern = re.compile(re.escape(old_str), re.IGNORECASE)
    new_content = pattern.sub(replacer, content)

    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(new_content)

def rename_files_in_directory(directory, old_str, new_str):
    """递归地重命名目录中的文件和文件夹，并替换文件内容中的字符串"""
    for root, dirs, files in os.walk(directory, topdown=False):
        # 先处理文件名
        for name in files:
            if old_str.lower() in name.lower():
                new_name = re.sub(re.escape(old_str), lambda m: ''.join(
                    new_str[i].upper() if c.isupper() else new_str[i].lower()
                    for i, c in enumerate(m.group(0))), name, flags=re.IGNORECASE)
                old_path = os.path.join(root, name)
                new_path = os.path.join(root, new_name)
                print(f'Renaming file: {old_path} -> {new_path}')
                os.rename(old_path, new_path)
                # replace_in_file(new_path, old_str, new_str)

        # 处理目录名
        for name in dirs:
            if old_str.lower() in name.lower():
                new_name = re.sub(re.escape(old_str), lambda m: ''.join(
                    new_str[i].upper() if c.isupper() else new_str[i].lower()
                    for i, c in enumerate(m.group(0))), name, flags=re.IGNORECASE)
                old_path = os.path.join(root, name)
                new_path = os.path.join(root, new_name)
                print(f'Renaming directory: {old_path} -> {new_path}')
                os.rename(old_path, new_path)

if __name__ == "__main__":
    directory = './mmcv/ops/csrc/common/musa'
    old_str = "cuh"
    new_str = "muh"

    rename_files_in_directory(directory, old_str, new_str)
    print("处理完成！")