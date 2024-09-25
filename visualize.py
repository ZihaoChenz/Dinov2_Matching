from UI.UI_interface import ImageDisplay
from utils.cal_similarity import compare_similarity
import tkinter as tk
from utils.txt_operation import generate_similarity_txt
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='parameter')
    parser.add_argument('--ImageType', help='choose the dataset images type', required=True, type=str)
    parser.add_argument('--ResultFolder', help='select result folder path', required=True, type=str)
    parser.add_argument('--DataFolder', help='select data folder path', required=True, type=str)
    parser.add_argument('--GenerateTxt', help='Generate all compare files similarity, choose output path', required=False, type=str)
    args = parser.parse_args()
    return args

args = parse_args()
image_type = args.ImageType
result_folder = args.ResultFolder
data_folder = args.DataFolder
txt_save = args.GenerateTxt

# visualize the UI interface
def visualize_result(data):
    root = tk.Tk()
    ImageDisplay(root, data, data_folder, centroid=False)
    root.mainloop()


if __name__ == '__main__':
    # get the total dict data
    if txt_save:
        data, total_dict = compare_similarity(image_type, result_folder)
        generate_similarity_txt(txt_save, total_dict)
    else:
        data = compare_similarity(image_type, result_folder)
    visualize_result(data)