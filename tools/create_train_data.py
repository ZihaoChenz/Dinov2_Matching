import os

# 输入的源文件夹路径
root_path = r'D:\Github-my\Dinov2\Dinov2_Matching\datasets\cityscape_stuttgart_00\val'
queries_folder = os.path.join(root_path, 'queries')

# 目标文件夹路径
positives_folder = os.path.join(root_path, 'positives')
negatives_folder = os.path.join(root_path, 'negatives')

# 检查并创建目标文件夹
if not os.path.exists(positives_folder):
    os.makedirs(positives_folder)
if not os.path.exists(negatives_folder):
    os.makedirs(negatives_folder)

# 获取queries文件夹中的所有文件
files = os.listdir(queries_folder)

# 遍历每个文件，创建子文件夹
for file in files:
    # 构造文件的完整路径
    file_path = os.path.join(queries_folder, file)

    # 检查文件是否是一个文件，而非子目录
    if os.path.isfile(file_path):
        # 构造子文件夹名称
        folder_name = os.path.splitext(file)[0]  # 去掉文件扩展名

        # 在positives和negatives文件夹中创建相应的子文件夹
        positive_folder = os.path.join(positives_folder, folder_name)
        # negative_folder = os.path.join(negatives_folder, folder_name)

        # 创建子文件夹（如果不存在的话）
        if not os.path.exists(positive_folder):
            os.makedirs(positive_folder)
            print(f"创建子文件夹 {positive_folder}")

        # if not os.path.exists(negative_folder):
        #     os.makedirs(negative_folder)
        #     print(f"创建子文件夹 {negative_folder}")

print("所有文件名的子文件夹已创建完成。")
