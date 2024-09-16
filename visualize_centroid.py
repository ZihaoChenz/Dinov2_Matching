from UI.UI_interface import ImageDisplay
from utils.cal_similarity import compare_centroid_similarity
import tkinter as tk
import argparse
from utils.load_data import load_centroids_data
from utils.cal_similarity import classify_centroids_cls

def parse_args():
    parser = argparse.ArgumentParser(description='parameter')
    parser.add_argument('--ImageType', help='choose the dataset images type', required=True, type=str)
    parser.add_argument('--GalleryData', help="path to root where previously prepared embeddings and cls were saved", required=True, type=str)
    parser.add_argument('--CheckTxtFolder', help='Txt folder path that to check. example: output/surrounding/xxx/check', required=True, type=str)
    parser.add_argument('--OutputBaseFolder', help='Output base folder, example: output/surrounding', required=True, type=str)
    parser.add_argument('--Normalize', help='choose whether normalize each class centroid tensor', action='store_true')
    args = parser.parse_args()
    return args
args = parse_args()
image_type = args.ImageType
gallery_data = args.GalleryData
check_txt_folder = args.CheckTxtFolder
base_folder = args.OutputBaseFolder

def visualize_result_controid(data):
    root = tk.Tk()
    ImageDisplay(root, data, data_folder=None, centroid=True)
    root.mainloop()

if __name__ == '__main__':
    centroid_cls_dict = load_centroids_data(gallery_data, normalize=args.Normalize)
    file_cls_dict = classify_centroids_cls(check_txt_folder, centroid_cls_dict)
    data = compare_centroid_similarity(image_type, base_folder, file_cls_dict)
    visualize_result_controid(data)