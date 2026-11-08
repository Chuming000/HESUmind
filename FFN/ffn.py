import torch
import torch.nn as nn
import torch.nn.functional as F

# ReLU FFN：经典 Transformer
class ReLUFFN(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()

        # 第一层：将隐藏维度 D 扩大到中间维度 M
        self.fc1 = nn.Linear(hidden_size, intermediate_size)

        # 第二层：从 M 映射回 D
        self.fc2 = nn.Linear(intermediate_size, hidden_size)

    def forward(self, x):
        # x: [Batch, Seq, D]

        x = self.fc1(x)  # [B, T, D] -> [B, T, M]
        x = F.relu(x)  # 非线性激活，负值置零
        x = self.fc2(x)  # [B, T, M] -> [B, T, D]

        return x
    
