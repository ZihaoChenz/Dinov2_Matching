import os
from PIL import Image

# 根据实际情况修改此处
negatives_dir = r"D:\Github-my\Dinov2\Dinov2_Matching\datasets\cityscape_stuttgart_00\train\negatives"  # negatives 文件夹的路径
images_dir = r"D:\Github-my\dataset\cityscapes(video)\leftImg8bit_demoVideo\leftImg8bit\demoVideo\stuttgart_00"  # 存放 png 图片的文件夹路径


def get_index_from_name(name: str) -> int:
    """
    从文件(夹)名中解析出编号部分。
    假设文件(夹)名格式为:
        stuttgart_00_000000_000XYZ_leftImg8bit
    或者
        stuttgart_00_000000_000XYZ_leftImg8bit.png
    等等...
    则可通过分割字符串来提取 XYZ，并转为 int。
    """
    parts = name.split("_")  # 例如 ["stuttgart","00","000000","000001","leftImg8bit"]
    # 第四个元素(parts[3])即为编号 "000001"
    # 如果带有 .png，则最后一个 parts[4] 可能是 "leftImg8bit.png"，但这不影响 parts[3]
    return int(parts[3])


def main():
    # 1) 遍历 negatives 文件夹，获取所有子文件夹
    for subfolder_name in os.listdir(negatives_dir):
        subfolder_path = os.path.join(negatives_dir, subfolder_name)

        # 只处理文件夹（跳过文件）
        if not os.path.isdir(subfolder_path):
            continue

        # 2) 从子文件夹名称中解析出该子文件夹的编号 X
        try:
            X = get_index_from_name(subfolder_name)
        except ValueError:
            # 如果无法解析到有效数字，跳过
            continue

        # 前后 10 个编号的区间
        lower_bound = X - 10
        upper_bound = X + 10

        # 3) 遍历图片文件夹中的所有 png 图片
        for img_name in os.listdir(images_dir):
            if not img_name.endswith(".png"):
                continue

            # 从图片名称中获取其编号 Y
            try:
                Y = get_index_from_name(img_name)
            except ValueError:
                # 如果无法解析到有效数字，跳过
                continue

            # 如果 Y 不在 [X-10, X+10] 范围内，则转换为 jpg 放入当前子文件夹
            if not (lower_bound <= Y <= upper_bound):
                img_path = os.path.join(images_dir, img_name)

                # 读取图片并转换
                with Image.open(img_path) as img:
                    # 输出文件名，把原来的 .png 换成 .jpg
                    out_name = img_name.replace(".png", ".jpg")
                    out_path = os.path.join(subfolder_path, out_name)

                    # 转换为 RGB 并保存为 JPG 格式
                    img.convert("RGB").save(out_path, "JPEG")

                print(f"Convert {img_name} -> {out_name} in {subfolder_name}")


if __name__ == "__main__":
    main()
