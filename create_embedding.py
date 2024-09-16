import argparse
from utils.embedding_process import create_embedding

def parse_args():
    parser = argparse.ArgumentParser(description='parameter')
    parser.add_argument('--Embedding_target_folder', help="if use centroid, choose txt target folder", required=False,
                        type=str)
    parser.add_argument('--Embedding_save_folder', help="if use centroid, choose path to save embedding",
                        required=False, type=str)
    args = parser.parse_args()
    return args

args = parse_args()




if __name__ == '__main__':
    target_folder = args.Embedding_target_folder
    save_folder = args.Embedding_save_folder
    create_embedding(target_folder, save_folder)