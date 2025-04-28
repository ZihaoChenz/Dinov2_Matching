import os.path
import json
from pathlib import Path
from UI.UI_interface import ImageDisplay
from utils.cal_similarity import compare_similarity
import tkinter as tk
from utils.txt_operation import generate_similarity_txt
import argparse
from evaluation import evaluate

def parse_args():
    parser = argparse.ArgumentParser(description='parameter')
    parser.add_argument('--ImageType', help='choose the dataset images type', required=True, type=str)
    parser.add_argument('--ResultFolder', help='select result folder path', required=True, type=str)
    parser.add_argument('--DataFolder', help='select data folder path', required=True, type=str)
    parser.add_argument('--json', help='Generate all compare files similarity, choose output path', required=False, type=str)
    parser.add_argument('--Eval', help='evaluate the dataset accuracy', required=False, action='store_true')
    args = parser.parse_args()
    return args

args = parse_args()
image_type = args.ImageType
result_folder = args.ResultFolder
data_folder = args.DataFolder
txt_save_folder = args.json
eval = args.Eval

# visualize the UI interface
def visualize_result(data):
    root = tk.Tk()
    ImageDisplay(root, data, data_folder, centroid=False)
    root.mainloop()


if __name__ == '__main__':
    # get the total dict data
    if txt_save_folder:
        if not os.path.exists(txt_save_folder):
            os.makedirs(txt_save_folder, exist_ok=True)
        top3_data, total_dict = compare_similarity(image_type, result_folder)
        json_save_path = Path(os.path.join(txt_save_folder, os.path.basename(result_folder) + ".json")).as_posix()
        with open(json_save_path, "w") as json_file:
            json.dump(total_dict, json_file, indent=4)
    else:
        top3_data, total_dict = compare_similarity(image_type, result_folder)
    # visualize_result(top3_data)
    if eval:
        accuracy = evaluate(total_dict)
        print("Dataset accuracy: ", accuracy)

