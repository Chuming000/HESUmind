'''
Layernorm & RMSNorm pytorch实现
'''
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim, eps:float=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
        
    def _norm(self, x):
        return x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        # x.float() 归一化时将 x 转成 FP32, 提高数值稳定性, 避免 FP16/BF16 计算平方、均值时出现数值精度问题。

    def forward(self, x):
        return (self.weight * self._norm(x)).to(x.dtype)
        # 输出转回原来的数据类型

class LayerNorm(nn.Module):
    def __init__(self, dim, eps:float=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps
        
    def _norm(self, x):
        x = x.float()
        mean = x.mean(-1, keepdim=True)
        var = (x-mean).pow(2).mean(-1, keepdim=True)
        return (x - mean) * torch.rsqrt(var + self.eps)
    
    def forward(self, x):
        return (self.weight * self._norm(x) + self.bias).to(x.dtype)
    
    
if __name__ == "__main__":

    x = torch.tensor([
        [1.0, 2.0, 3.0],
        [11.0, 12.0, 13.0]
    ])

    rms_norm = RMSNorm(dim=3)
    layer_norm = LayerNorm(dim=3)

    print("原始数据：")
    print(x)

    print("\nRMSNorm 输出：")
    print(rms_norm(x).detach())

    print("\nLayerNorm 输出：")
    print(layer_norm(x).detach())
    
    """
    LayerNorm：消除整体偏移（均值），同时调整尺度。
    RMSNorm：只按均方根调整尺度，不消除整体偏移。
    """
    