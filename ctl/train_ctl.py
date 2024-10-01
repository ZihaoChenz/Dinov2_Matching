import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from model import model_process
from models.model_tune import FineTuneModel

# load the base model
base_model = model_process()
# load the fine tune model
fine_tune_model = FineTuneModel(base_model)
# Define the optimizer (only the parameters of the last two layers will be optimized)
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, fine_tune_model.parameters()), lr=1e-4)


