import torch
import torch.nn as nn
from utils.cal_similarity import compute_similarity
import torch.nn.functional as F

class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, device, temperature=0.07):
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.device = device
    def forward(self, features, pos_indices, num_samples_per_query):
        """
        在变量后使用clone确保不会进行原地操作导致在反向传播时报错
        """
        batch_size = len(pos_indices)
        total_samples = features.size(0)

        norms = torch.norm(features.clone(), p=2, dim=1, keepdim=True)
        norms = norms.clone()
        norms[norms == 0] = 1
        features_normalized = features.clone() / norms
        cos_similarity_matrix = torch.matmul(features_normalized, features_normalized.T)

        # 创建掩码矩阵
        mask = torch.zeros((total_samples, total_samples)).to(self.device)
        for i in range(batch_size):
            query_idx = i * num_samples_per_query
            pos_idxs = pos_indices[i]
            for pos_idx in pos_idxs:
                mask[query_idx, pos_idx] = 1    # 查询图像与正样本
                mask[pos_idx, query_idx] = 1    # 正样本与查询图像
        mask = mask.clone()

        # 去除自身对比
        # logits_mask = torch.ones_like(mask) - torch.eye(total_samples).to(self.device)
        logits_mask = (torch.ones_like(mask) - torch.eye(total_samples, device=self.device)).clone()
        mask = mask * logits_mask.clone()

        # 应用温度缩放
        logits = cos_similarity_matrix / self.temperature

        # 计算对数概率
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        # 计算平均正样本对数概率
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-12)

        # 计算损失
        loss = -mean_log_prob_pos
        loss = loss.mean()

        return loss

