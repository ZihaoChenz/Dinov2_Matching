import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets
import torchvision.transforms as transforms
from pathlib import Path
from PIL import Image


# #自定义数据集包装器
class SubsetWithFilenames(Dataset):
    def __init__(self, subset):
        self.subset = subset
        self.dataset = subset.dataset  # 获取原始数据集

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        image, label = self.subset[idx]  # 获取图像和标签
        # 通过索引获取文件路径
        path, _ = self.dataset.samples[self.subset.indices[idx]]
        filename = os.path.basename(path)  # 获取文件名
        return image, label, filename


class QueryPositiveDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        参数：
        - root_dir：数据集的根目录，例如 ‘dataset/train’
        - transform：默认不进行数据预处理操作
        """
        self.root_dir = root_dir
        self.transform = transform

        # 查询图像目录
        self.query_dir = os.path.join(root_dir, 'queries')
        # 正样本目录
        self.positive_dir = os.path.join(root_dir, 'positives')

        # 负样本目录
        self.negative_dir = os.path.join(root_dir, 'negatives')

        # 获取所有查询图像的文件名列表
        self.query_filenames = sorted(os.listdir(self.query_dir))

    def __len__(self):
        # 查询图像的数量
        return len(self.query_filenames)

    def __getitem__(self, idx):
        # 获取每个查询图像的路径
        query_filename = self.query_filenames[idx]
        query_path = os.path.join(self.query_dir, query_filename)

        # 加载查询图像
        query_image = Image.open(query_path).convert('RGB')

        if self.transform:
            query_image = self.transform(query_image)

        # 根据查询图像的文件名，找到对应的正样本文件夹
        query_name = os.path.splitext(query_filename)[0]
        positive_folder = os.path.join(self.positive_dir, query_name)
        # 找到对应的负样本文件夹
        negative_folder = os.path.join(self.negative_dir, query_name)

        # 获取该查询图像对应的所有正样本的图像的文件名
        positive_filenames = sorted(os.listdir(positive_folder))
        # 获取该查询图像对应的所有负样本的图像的文件名
        negative_filenames = sorted(os.listdir(negative_folder))


        # 加载所有的负样本图像
        negative_image = []
        for neg_filename in negative_filenames:
            neg_path = os.path.join(negative_folder, neg_filename)
            neg_image = Image.open(neg_path).convert('RGB')
            if self.transform:
                neg_image = self.transform(neg_image)
            negative_image.append(neg_image)

        # 加载所有的正样本图像
        positive_images = []
        for pos_filename in positive_filenames:
            pos_path = os.path.join(positive_folder, pos_filename)
            pos_image = Image.open(pos_path).convert('RGB')
            if self.transform:
                pos_image = self.transform(pos_image)
            positive_images.append(pos_image)

        return query_image, positive_images, negative_image


def custom_collate_fn(batch):
    """
    自定义collate_fn函数，将列表的列表转换为张量
    """
    queries = []
    positives = []
    negatives = []

    for item in batch:
        query_image, positive_images, negative_images = item
        queries.append(query_image)
        # 将正负样本转换为张量并且放进列表中
        positives.append(torch.stack(positive_images))
        negatives.append(torch.stack(negative_images))

    # 将查询图像转换为张量
    queries = torch.stack(queries)
    # 将正负样本堆叠成一个张量
    positives = torch.stack(positives) # 形状为 [batch_size, 4, C, H, W]
    negatives = torch.stack(negatives)

    return queries, positives, negatives


# Define a function for loading and transforming image data
def load_data(check_folder):
    # Define transformations: random crop, random flip, convert to tensor, and normalize
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),  # Resize and crop the image to a 224x224 square
        transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
        transforms.ToTensor(),  # Convert the image to a tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize the image with mean and standard deviation
    ])

    # Load the dataset from directory and apply transformations
    full_dataset = datasets.ImageFolder(check_folder, transform)
    # 获取类别到索引的映射
    class_to_idx = full_dataset.class_to_idx

    # 创建字典来存储不同类别的数据集
    subsets = {}

    # 遍历每个类别
    for class_name, class_index in class_to_idx.items():
        # 获取属于当前类别的所有样本的索引
        indices = [i for i, (_, label) in enumerate(full_dataset.imgs) if label == class_index]

        # 使用 Subset 创建当前类别的子集
        subsets[class_name] = Subset(full_dataset, indices)

    check_with_filenames = SubsetWithFilenames(subsets['check'])
    ref_with_filenames = SubsetWithFilenames(subsets['ref'])

    check_loader = DataLoader(check_with_filenames, batch_size=1, shuffle=True)  # Shuffle the training data
    ref_loader = DataLoader(ref_with_filenames, batch_size=1, shuffle=False)  # No need to shuffle validation data

    return {'check': check_loader, 'ref': ref_loader} # Return loaders and number of classes in the dataset

def load_data_train(dataset_root_dir, batch_size):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 调整图像大小
        transforms.ToTensor(),  # 转为Tensor格式
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 图像归一化
    ])

    # 创建train的dataloader
    train_path = os.path.join(dataset_root_dir, 'train')
    train_dataset = QueryPositiveDataset(root_dir=train_path, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)

    # 创建val的dataloader
    val_path = os.path.join(dataset_root_dir, 'val')
    val_dataset = QueryPositiveDataset(root_dir=val_path, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)


    return train_loader, val_loader


def load_centroids_data(gallery_data, normalize):
    # load cls.npy and embeddings.npy data in gallery data path
    LOAD_PATH = Path(gallery_data)
    embeddings_gallery = torch.from_numpy(
        np.load(LOAD_PATH / "embeddings.npy", allow_pickle=True)
    )
    paths_gallery = np.load(LOAD_PATH / "cls.npy", allow_pickle=True)

    # Normalize all tensor data if needed
    if normalize:
        embeddings_gallery = torch.nn.functional.normalize(
            embeddings_gallery, dim=1, p=2
        )

    # Create a dict that maps all class names to their centroid tensors
    centroid_data_dict = {key: value for key, value in zip(paths_gallery, embeddings_gallery)}

    return centroid_data_dict

# For testing loading data
if __name__ == '__main__':
    load_centroids_data(r"D:\GitHub_my\Dinov2\Dinov2_Matching\embedding\surrounding", normalize=False)