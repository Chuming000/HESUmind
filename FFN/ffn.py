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


# SwiGLU FFN：LLaMA / PaLM 等模型采用的门控前馈网络
#
# 与 ReLU FFN 的区别：
#   1. 有三条投影而不是两条（gate / up / down），多出来的是「门控」分支
#   2. 激活函数用 SiLU（Swish），并且只作用在 gate 分支上
#   3. 两个分支逐元素相乘后再降维，等于用 gate 动态缩放 up 的每个通道
#   4. 默认不带 bias（LLaMA 系的做法）
class SwiGLUFFN(nn.Module):
    def __init__(self, hidden_size, intermediate_size=None, multiple_of=256, bias=False):
        super().__init__()

        if intermediate_size is None:
            # 门控结构多了一条投影，参数量约为 ReLU FFN 的 1.5 倍，
            # 因此把中间维度从 4D 缩到 8/3D，使两者参数量大致相当，
            # 再向上对齐到 multiple_of 的整数倍（LLaMA-2 里 D=4096 -> 11008）
            intermediate_size = int(8 * hidden_size / 3)
            intermediate_size = multiple_of * ((intermediate_size + multiple_of - 1) // multiple_of)

        self.intermediate_size = intermediate_size

        # 门控分支：决定每个通道「放行多少」
        self.w_gate = nn.Linear(hidden_size, intermediate_size, bias=bias)

        # 值分支：提供实际内容
        self.w_up = nn.Linear(hidden_size, intermediate_size, bias=bias)

        # 输出投影：从 M 映射回 D
        self.w_down = nn.Linear(intermediate_size, hidden_size, bias=bias)

    def forward(self, x):
        # x: [Batch, Seq, D]

        gate = F.silu(self.w_gate(x))  # [B, T, D] -> [B, T, M]，SiLU 起门控作用
        up = self.w_up(x)  # [B, T, D] -> [B, T, M]，内容分支不做激活

        x = gate * up  # 逐元素相乘：用 gate 缩放 up 的每个通道
        x = self.w_down(x)  # [B, T, M] -> [B, T, D]

        return x


if __name__ == "__main__":
    torch.manual_seed(0)

    B, T, D = 2, 8, 512
    x = torch.randn(B, T, D)

    relu_ffn = ReLUFFN(D, 4 * D)
    swiglu_ffn = SwiGLUFFN(D)

    def count(m):
        return sum(p.numel() for p in m.parameters())

    print(f"输入: {tuple(x.shape)}\n")
    print(f"{'模块':<12}{'中间维度':>10}{'参数量':>12}{'输出形状':>16}")
    print("-" * 52)
    for name, m in (("ReLUFFN", relu_ffn), ("SwiGLUFFN", swiglu_ffn)):
        y = m(x)
        print(f"{name:<12}{m.intermediate_size if hasattr(m, 'intermediate_size') else 4 * D:>10}"
              f"{count(m):>12,}{str(tuple(y.shape)):>16}")

    ratio = count(swiglu_ffn) / count(relu_ffn)
    print(f"\n参数量比值 SwiGLU / ReLU = {ratio:.3f}")
    print("对齐到 256 的整数倍会让小模型的比值偏大；D 越大越接近 1.0。")
