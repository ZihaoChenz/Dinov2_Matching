import torch
import torch.nn as nn
import torch.nn.functional as F

class FineTuneModel(nn.Module):
    def __init__(self, base_model):
        super(FineTuneModel, self).__init__()
        # 使用预训练的base_model并去掉最后的全连接层
        self.base = nn.Sequential(*list(base_model.children())[:-2])
        # 添加新的全连接层
        self.fc1 = nn.Linear(base_model.fc.in_features, 512)
        self.fc2 = nn.Linear(512, 384)  # 用于特征提取

        # 冻结除最后两层外的所有参数
        self.freeze_base_layers()

    def freeze_base_layers(self):
        """
        冻结除最后两层外的所有参数，避免这些层在训练过程中被更新。
        """
        for name, param in self.named_parameters():
            if "fc1" not in name and "fc2" not in name:  # 只保留fc1和fc2的梯度更新
                param.requires_grad = False

    def forward(self, x):
        x = self.base(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.normalize(self.fc2(x), p=2, dim=1)  # L2 归一化
        return x


# 使用方法
# # 使用预训练的 ResNet50 作为基础模型
# base_model = models.resnet50(pretrained=True)
#
# # 实例化 FineTuneModel，自动冻结除最后两层外的参数
# model = FineTuneModel(base_model)
#
# # 定义优化器（只会优化最后两层的参数）
# optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

