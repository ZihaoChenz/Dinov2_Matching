import torch
import os
import torch.nn as nn
from torchvision import models
import argparse
from model import model_process
from ctl.ctl_loss import SupervisedContrastiveLoss
from utils.load_data import load_data_train
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description='parameter')
    parser.add_argument('--DataFolder', help="input dataset folder", required=True, type=str)
    parser.add_argument('--CheckPoints', help="output model folder path", required=False, type=str, default='checkpoints')
    parser.add_argument('--batch_size', help='query image number in each batch', required=True, type=int)
    # parser.add_argument('--num_pos', help='the number of positive image of each query image', required=True, type=int)
    parser.add_argument('--epoch', help='number of epoch', required=True, type=int)
    args = parser.parse_args()
    return args

args = parse_args()
dataset = args.DataFolder
checkpoint_folder = args.CheckPoints
if not os.path.exists(checkpoint_folder):
    os.makedirs(checkpoint_folder)
batch_size = args.batch_size
# num_pos = args.num_pos
num_epochs = args.epoch


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


model = model_process() # this will load the small model
# model = model_process(backbone = 'dinov2_b') # to load the base model
# model = model_process(backbone = 'dinov2_l') # to load the large model
# model = model_process(backbone = 'dinov2_g') # to load the largest model

model.to(device)

# 冻结其他层的参数，只finetune全连接层
for name, param in model.named_parameters():
    # param.requires_grad = True
    # 检查层名是否包含 'fc1' 或 'fc2'，如果包含则将 requires_grad 设为 True，否则设为 False
    if 'blocks.11' in name or 'blocks.11' in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

# print(model)


# 只更新最后一个block的两个全连接层的参数
optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

# # 打印优化器中的参数
# for group in optimizer.param_groups:
#     for p in group['params']:
#         print(p.shape, p.requires_grad)

criterion = SupervisedContrastiveLoss(device=device)

train_loader, val_loader = load_data_train(dataset, batch_size=batch_size)

# 用于记录每个epoch的loss
epoch_losses = []

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0  # 每个epoch的总损失

    for batch_idx, (queries, positives, negatives) in enumerate(train_loader):
        batch_size = queries.size(0)
        # 每个query的正样本数量
        num_pos = positives.shape[1]
        # 每个query的负样本数量
        num_neg = negatives.shape[1]
        # 每个query的正负样本数量（包括query本身）
        num_samples_per_query = num_pos + num_neg + 1  # 包括查询、正样本和负样本




        queries = queries.to(device)
        positives = positives.to(device)
        negatives = negatives.to(device)

        # 展开正负样本
        positives = positives.view(batch_size * num_pos, positives.size(2), positives.size(3), positives.size(4))
        negatives = negatives.view(batch_size * num_neg, negatives.size(2), negatives.size(3), negatives.size(4))


        # 前向传播
        query_features = model(queries)  # [batch_size, feature_dim]
        positive_features = model(positives)  # [batch_size * K, feature_dim]
        negative_features = model(negatives)
        positive_features = positive_features.view(batch_size, num_pos, -1)  # [batch_size, num_pos, feature_dim]
        negative_features = negative_features.view(batch_size, num_neg, -1)

        # print(query_features.requires_grad)  # 应该为 True
        # print(positive_features.requires_grad)  # 应该为 True

        # 合并特征，包括查询、正样本和负样本
        all_features = torch.cat([query_features.unsqueeze(1), positive_features, negative_features], dim=1)  # [batch_size, num_pos + num_neg + 1, feature_dim]
        all_features = all_features.view(-1, all_features.size(-1))  # [batch_size * num_samples_per_query, feature_dim]

        # 构建正样本索引列表
        pos_indices = []
        for i in range(batch_size):
            query_idx = i * num_samples_per_query
            positive_idxs = [query_idx + j + 1 for j in range(num_pos)]  # 正样本索引紧跟在查询之后
            pos_indices.append(positive_idxs)

        # 计算损失```````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````
        loss = criterion(all_features, pos_indices, num_samples_per_query)
        # print(loss.grad_fn)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()  # 累积当前batch的损失

        if batch_idx % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")

    # 记录当前epoch的平均损失
    avg_epoch_loss = epoch_loss / len(train_loader)
    epoch_losses.append(avg_epoch_loss)

    model_path = os.path.join(checkpoint_folder, f'model_epoch_{epoch + 1}.pth')
    # torch.save(model.state_dict(), model_path)
    print(f"模型已保存：{model_path}")

# 绘制损失曲线
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), epoch_losses, label="Train Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.legend()
plt.grid(True)
plt.show()