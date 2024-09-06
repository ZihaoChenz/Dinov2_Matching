import os
import torch
import torch.nn.functional as F
from utils.txt_operation import load_txt_to_tensor
import tkinter as tk
from PIL import Image, ImageTk
from UI.UI_interface import ImageDisplay

import argparse
def parse_args():
    parser = argparse.ArgumentParser(description='parameter')
    parser.add_argument('--ImageType', help='choose the dataset images type', required=True, type=str)
    args = parser.parse_args()
    return args
args = parse_args()
image_type = args.ImageType


def compute_similarity(feature1, feature2, method='cosine'):
    """
    计算两个图片特征之间的相似性。

    参数:
    feature1 (torch.Tensor): 第一张图片的特征向量。
    feature2 (torch.Tensor): 第二张图片的特征向量。
    method (str): 计算相似度的方法。可选 'cosine' 或 'euclidean'。默认是 'cosine'。

    返回:
    float: 两个特征之间的相似性值。
    """
    if method == 'cosine':
        # 使用PyTorch的余弦相似度计算
        similarity = F.cosine_similarity(feature1, feature2, dim=1)
        return similarity.item()

    elif method == 'euclidean':
        # 计算欧氏距离
        distance = torch.dist(feature1, feature2, p=2)  # p=2 代表欧氏距离
        return distance.item()

    else:
        raise ValueError("不支持的计算方法。请使用 'cosine' 或 'euclidean'")

def compare_similarity():
    check_txt = os.listdir('../output/check')
    ref_txt = os.listdir('../output/ref')
    total_dict = {}
    for c_txt in check_txt:
        similarity_dict = {}
        for r_txt in ref_txt:
            feature_c = load_txt_to_tensor(os.path.join('../output/check', c_txt))
            feature_r = load_txt_to_tensor(os.path.join('../output/ref', r_txt))
            similarity_dict[r_txt.split('.')[0] + '.' + image_type] = compute_similarity(feature_c, feature_r)
        top_three = dict(sorted(similarity_dict.items(), key=lambda item: item[1], reverse=True)[:3])
        total_dict[c_txt.split('.')[0] + '.' + image_type] = top_three
    return total_dict

def visualize_result(data):
    root = tk.Tk()
    ImageDisplay(root, data)
    root.mainloop()

if __name__ == '__main__':
    # feature_1 = load_txt_to_tensor('D:\\Github_Project\\Downstream-Dinov2\\output\\check\\IMG_4287.txt')
    # feature_2 = load_txt_to_tensor('D:\\Github_Project\\Downstream-Dinov2\\output\\check\\IMG_4288.txt')
    # print(feature_1)
    # print(feature_2)
    # similarity = compute_similarity(feature_1, feature_2, method='cosine')
    # print("similarity: ", similarity)
    data = compare_similarity()
    visualize_result(data)