import os

def check_file_counts(directory):
    # 存储每个子文件夹的文件数量
    folder_counts = {}

    # 遍历指定文件夹中的所有子文件夹
    for root, dirs, files in os.walk(directory):
        # 只关注直接的子文件夹，而不是递归子文件夹
        if root == directory:
            for dir_name in dirs:
                subfolder_path = os.path.join(root, dir_name)
                file_count = sum([1 for f in os.listdir(subfolder_path) if os.path.isfile(os.path.join(subfolder_path, f))])
                folder_counts[subfolder_path] = file_count

    # 找出文件数量不一致的子文件夹
    count_values = list(folder_counts.values())
    if len(set(count_values)) > 1:
        print("文件数量不一致的子文件夹:")
        for folder, count in folder_counts.items():
            print(f"{folder}: {count} 个文件")
    else:
        print("所有子文件夹的文件数量一致.")

# 替换成你要检查的文件夹路径
directory_to_check = r"D:\Github-my\Dinov2\Dinov2_Matching\datasets\cityscape\val\negatives"
check_file_counts(directory_to_check)
