import os
import subprocess


def run_inference_ZuBuD(src_folder):
    # 获取源文件夹中的所有objectXXXX文件夹
    for object_folder in os.listdir(src_folder):
        object_folder_path = os.path.join(src_folder, object_folder)

        # 确认这是一个文件夹，并且符合 objectXXXX 的命名规则
        if os.path.isdir(object_folder_path) and object_folder.startswith("object"):
            print(f"Processing folder: {object_folder_path}")

            # 构建CheckFolder和OutputFolder的路径
            check_folder = os.path.join("../data/ZuBuD", object_folder)
            output_folder_path = os.path.join("../output/ZuBuD", object_folder)

            # 确保 OutputFolder 路径存在，如果不存在则创建它
            if not os.path.exists(output_folder_path):
                os.makedirs(output_folder_path)

            # 调用 inference.py，并传递 CheckFolder 和 OutputFolder 参数
            try:
                subprocess.run(
                    ['python', '../inference.py', '--CheckFolder', check_folder, '--OutputFolder', output_folder_path],
                    check=True)
                print(f"Finished processing {object_folder}")
            except subprocess.CalledProcessError as e:
                print(f"Error processing {object_folder}: {e}")


# 使用示例，指定 objectXXXX 文件夹的上一级路径以及输出文件夹路径
src_folder = r"D:\Github-my\Dinov2\Dinov2_Matching\data\ZuBuD"  # 包含objectXXXX的文件夹路径
run_inference_ZuBuD(src_folder)
