import torch
import torch.nn as nn
import torch.nn.functional as F

class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, device, temperature):
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.device = device

    def forward(self, features, pos_indices, num_samples_per_query):
        batch_size = len(pos_indices)
        total_samples = features.size(0)

        # Normalize feature embeddings
        features_normalized = F.normalize(features, p=2, dim=1)

        # Compute cosine similarity
        cos_similarity_matrix = torch.matmul(features_normalized, features_normalized.T)

        # Create mask matrix
        query_indices = torch.arange(0, batch_size * num_samples_per_query, num_samples_per_query).to(self.device)
        mask = torch.zeros((total_samples, total_samples), device=self.device)
        mask[query_indices[:, None], pos_indices] = 1
        mask[pos_indices, query_indices[:, None]] = 1

        # Remove self-contrast
        logits_mask = torch.ones_like(mask) - torch.eye(total_samples, device=self.device)
        mask = mask * logits_mask

        # Apply temperature scaling
        logits = cos_similarity_matrix / self.temperature

        # Compute log probability
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        # Compute weighted mean log probability of positives
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-12)

        # Compute loss
        loss = -mean_log_prob_pos.mean()

        return loss
