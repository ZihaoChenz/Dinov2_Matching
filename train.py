import torch
import os
import torch.nn as nn
from torchvision import models
import argparse
from model import model_process
import torch.nn.functional as F
from ctl.ctl_loss import SupervisedContrastiveLoss
from ctl.ctl_loss_2024_11_21 import SupervisedContrastiveLoss
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
# optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

# # 打印优化器中的参数
# for group in optimizer.param_groups:
#     for p in group['params']:
#         print(p.shape, p.requires_grad)

criterion = SupervisedContrastiveLoss(device=device)

train_loader, val_loader = load_data_train(dataset, batch_size=batch_size)

# 用于记录每个epoch的loss
# 记录每个epoch的训练和验证损失
train_losses = []
val_losses = []
train_pos_similarities = []
train_neg_similarities = []
val_pos_similarities = []
val_neg_similarities = []

for epoch in range(num_epochs):
    model.train()
    train_loss = 0

    # 初始化train累加器
    total_train_pos_similarity = 0.0
    total_train_neg_similarity = 0.0
    total_train_pos_count = 0
    total_train_neg_count = 0

    for batch_idx, (queries, positives, negatives) in enumerate(train_loader):
        batch_size = queries.size(0)
        num_pos = positives.shape[1]
        num_neg = negatives.shape[1]
        num_samples_per_query = num_pos + num_neg + 1

        queries = queries.to(device)
        positives = positives.to(device)
        negatives = negatives.to(device)

        positives = positives.view(batch_size * num_pos, positives.size(2), positives.size(3), positives.size(4))
        negatives = negatives.view(batch_size * num_neg, negatives.size(2), negatives.size(3), negatives.size(4))

        # 前向传播
        query_features = model(queries)
        positive_features = model(positives)
        negative_features = model(negatives)
        positive_features = positive_features.view(batch_size, num_pos, -1)
        negative_features = negative_features.view(batch_size, num_neg, -1)

        # 归一化特征向量
        query_features_normalized = F.normalize(query_features, p=2, dim=1)
        positive_features_normalized = F.normalize(positive_features, p=2, dim=2)
        negative_features_normalized = F.normalize(negative_features, p=2, dim=2)

        # 计算正样本相似度
        positive_similarities = torch.bmm(positive_features_normalized, query_features_normalized.unsqueeze(2)).squeeze(2)  # shape (batch_size, num_pos)
        # 计算负样本相似度
        negative_similarities = torch.bmm(negative_features_normalized, query_features_normalized.unsqueeze(2)).squeeze(2)  # shape (batch_size, num_neg)

        # 累加相似度和计数
        total_train_pos_similarity += positive_similarities.sum().item()
        total_train_neg_similarity += negative_similarities.sum().item()
        total_train_pos_count += positive_similarities.numel()
        total_train_neg_count += negative_similarities.numel()

        all_features = torch.cat([query_features.unsqueeze(1), positive_features, negative_features], dim=1)
        all_features = all_features.view(-1, all_features.size(-1))

        pos_indices = []
        for i in range(batch_size):
            query_idx = i * num_samples_per_query
            positive_idxs = [query_idx + j + 1 for j in range(num_pos)]
            pos_indices.append(positive_idxs)

        loss = criterion(all_features, pos_indices, num_samples_per_query)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        if batch_idx % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], Train Loss: {loss.item():.4f}")

    avg_train_loss = train_loss / len(train_loader)
    train_losses.append(avg_train_loss)


    model.eval()
    val_loss = 0

    # 初始化val累加器
    total_val_pos_similarity = 0.0
    total_val_neg_similarity = 0.0
    total_val_pos_count = 0
    total_val_neg_count = 0

    with torch.no_grad():
        for val_batch_idx, (val_queries, val_positives, val_negatives) in enumerate(val_loader):
            batch_size = val_queries.size(0)
            num_pos = val_positives.shape[1]
            num_neg = val_negatives.shape[1]
            num_samples_per_query = num_pos + num_neg + 1

            val_queries = val_queries.to(device)
            val_positives = val_positives.to(device)
            val_negatives = val_negatives.to(device)

            val_positives = val_positives.view(batch_size * num_pos, val_positives.size(2), val_positives.size(3), val_positives.size(4))
            val_negatives = val_negatives.view(batch_size * num_neg, val_negatives.size(2), val_negatives.size(3), val_negatives.size(4))

            query_features = model(val_queries)
            positive_features = model(val_positives)
            negative_features = model(val_negatives)
            positive_features = positive_features.view(batch_size, num_pos, -1)
            negative_features = negative_features.view(batch_size, num_neg, -1)

            # 归一化特征向量
            query_features_normalized = F.normalize(query_features, p=2, dim=1)
            positive_features_normalized = F.normalize(positive_features, p=2, dim=2)
            negative_features_normalized = F.normalize(negative_features, p=2, dim=2)

            # 计算正样本相似度
            positive_similarities = torch.bmm(positive_features_normalized, query_features_normalized.unsqueeze(2)).squeeze(2)  # shape (batch_size, num_pos)
            # 计算负样本相似度
            negative_similarities = torch.bmm(negative_features_normalized, query_features_normalized.unsqueeze(2)).squeeze(2)  # shape (batch_size, num_neg)

            # 累加相似度和计数
            total_val_pos_similarity += positive_similarities.sum().item()
            total_val_neg_similarity += negative_similarities.sum().item()
            total_val_pos_count += positive_similarities.numel()
            total_val_neg_count += negative_similarities.numel()


            all_features = torch.cat([query_features.unsqueeze(1), positive_features, negative_features], dim=1)
            all_features = all_features.view(-1, all_features.size(-1))

            pos_indices = []
            for i in range(batch_size):
                query_idx = i * num_samples_per_query
                positive_idxs = [query_idx + j + 1 for j in range(num_pos)]
                pos_indices.append(positive_idxs)

            loss = criterion(all_features, pos_indices, num_samples_per_query)
            val_loss += loss.item()

            if val_batch_idx % 10 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{val_batch_idx + 1}/{len(train_loader)}], Val Loss: {loss.item():.4f}")

    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    # 计算平均相似度
    avg_train_pos_similarity = total_train_pos_similarity / total_train_pos_count
    avg_train_neg_similarity = total_train_neg_similarity / total_train_neg_count
    avg_val_pos_similarity = total_val_pos_similarity / total_val_pos_count
    avg_val_neg_similarity = total_val_neg_similarity / total_val_neg_count

    # 将平均相似度添加到列表中
    train_pos_similarities.append(avg_train_pos_similarity)
    train_neg_similarities.append(avg_train_neg_similarity)
    val_pos_similarities.append(avg_val_pos_similarity)
    val_neg_similarities.append(avg_val_neg_similarity)
    # print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    print(f"Epoch [{epoch + 1}/{num_epochs}], "
          f"Train Loss: {avg_train_loss:.4f}, "
          f"Val Loss: {avg_val_loss:.4f}, "
          f"Avg Train Pos Similarity: {avg_train_pos_similarity:.4f}, "
          f"Avg Train Neg Similarity: {avg_train_neg_similarity:.4f}, "
          f"Avg Val Pos Similarity: {avg_val_pos_similarity:.4f}",
          f"Avg Val Neg Similarity: {avg_val_neg_similarity:.4f}"
          )
    if epoch % 10 == 0:
        model_path = os.path.join(checkpoint_folder, f'model_epoch_{epoch + 1}.pth')
        torch.save(model, model_path)
        print(f"模型已保存：{model_path}")

# # 绘制训练和验证损失曲线
# plt.figure(figsize=(10, 6))
# plt.plot(range(1, num_epochs + 1), train_losses, label="Train Loss")
# plt.plot(range(1, num_epochs + 1), val_losses, label="Validation Loss")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.title("Training and Validation Loss Curve")
# plt.legend()
# plt.grid(True)
# plt.show(savefig="loss_image")

if __name__ == '__main__':
    # Save all the plots
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label="Train Loss")
    plt.plot(range(1, num_epochs + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Curve")
    plt.legend()
    plt.grid(True)
    loss_plot_path = os.path.join(checkpoint_folder, "loss_curve.png")
    plt.savefig(loss_plot_path)

    # Plot similarities
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), train_pos_similarities, label="Avg Train Pos Similarity")
    plt.plot(range(1, num_epochs + 1), train_neg_similarities, label="Avg Train Neg Similarity")
    plt.xlabel("Epoch")
    plt.ylabel("Similarity")
    plt.title("Training Similarity")
    plt.legend()
    plt.grid(True)
    train_similarity_plot_path = os.path.join(checkpoint_folder, "train_similarity.png")
    plt.savefig(train_similarity_plot_path)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), val_pos_similarities, label="Avg Val Pos Similarity")
    plt.plot(range(1, num_epochs + 1), val_neg_similarities, label="Avg Val Neg Similarity")
    plt.xlabel("Epoch")
    plt.ylabel("Similarity")
    plt.title("Validation Similarity")
    plt.legend()
    plt.grid(True)
    val_similarity_plot_path = os.path.join(checkpoint_folder, "val_similarity.png")
    plt.savefig(val_similarity_plot_path)
