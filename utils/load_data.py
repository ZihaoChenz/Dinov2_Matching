import os
import numpy as np
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

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