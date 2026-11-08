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


# MoE FFN：把一个大 FFN 拆成 N 个专家，每个 token 只走其中 top_k 个
#
# 与前面两个 FFN 的区别：
#   1. 参数量是「总量」而不是「每 token 计算量」——MoE 靠稀疏激活换容量
#   2. 多一个 router（门控网络），负责给每个 token 选专家
#   3. 需要负载均衡损失，否则 router 会塌缩到少数几个专家上
#   4. 专家的中间维度通常与稠密 FFN 持平（Mixtral 的做法），所以总参数量约等于 N 倍
class MoEFFN(nn.Module):
    def __init__(self, hidden_size, num_experts=8, top_k=2, expert_hidden=None,
                 multiple_of=256, bias=False):
        super().__init__()

        if top_k > num_experts:
            raise ValueError(f"top_k({top_k}) 不能大于专家数({num_experts})")

        self.num_experts = num_experts
        self.top_k = top_k

        # 路由网络：给每个 token 打 N 个分，不带 bias（与 LLaMA 系 router 一致）
        self.router = nn.Linear(hidden_size, num_experts, bias=False)

        # 专家本身就是一个普通的 SwiGLU FFN
        self.experts = nn.ModuleList(
            [SwiGLUFFN(hidden_size, expert_hidden, multiple_of, bias)
             for _ in range(num_experts)]
        )
        self.expert_hidden = self.experts[0].intermediate_size

        # 训练时由 forward 写入，供 loss 使用
        self.aux_loss = None

    def forward(self, x):
        # x: [Batch, Seq, D]
        B, T, D = x.shape
        flat = x.reshape(B * T, D)

        logits = self.router(flat)  # [N, E]
        probs = F.softmax(logits, dim=-1)  # 每个 token 对 E 个专家的偏好
        top_p, top_i = torch.topk(probs, self.top_k, dim=-1)  # [N, k]

        # 归一化被选中的权重，使其和为 1（LLaMA 系的做法）
        top_p = top_p / top_p.sum(dim=-1, keepdim=True)

        out = torch.zeros_like(flat)
        for e, expert in enumerate(self.experts):
            # 找出把专家 e 选进 top-k 的 (token, 槽位) 对
            hit = (top_i == e)  # [N, k]
            tok, slot = hit.nonzero(as_tuple=True)
            if tok.numel() == 0:  # 这个专家本轮没人选，直接跳过，省算力
                continue

            w = top_p[tok, slot]  # [M]，每个命中 token 分到的门控权重
            y = expert(flat[tok])  # 只把命中的 token 送进这个专家
            out.index_add_(0, tok, y * w.unsqueeze(-1))

        self.aux_loss = self._aux_loss(probs, top_i)
        return out.view(B, T, D)

    def _aux_loss(self, probs, top_i):
        """Switch Transformer 的负载均衡损失：E * sum(f_i * P_i)。

        f_i = 分给专家 i 的 token 占比，P_i = 专家 i 的平均路由概率。
        两者都均匀（各 1/E）时取最小值 1；全部挤到一个专家上会显著变大。
        """
        E = self.num_experts
        counts = torch.zeros(E, device=probs.device)
        ones = torch.ones(top_i.numel(), device=probs.device)
        counts.index_add_(0, top_i.reshape(-1), ones)

        f = counts / counts.sum()  # 实际分配比例
        P = probs.mean(dim=0)  # 平均路由概率
        return E * (f * P).sum()


if __name__ == "__main__":
    torch.manual_seed(0)

    B, T, D = 2, 8, 512
    x = torch.randn(B, T, D)

    relu_ffn = ReLUFFN(D, 4 * D)
    swiglu_ffn = SwiGLUFFN(D)
    moe_ffn = MoEFFN(D, num_experts=8, top_k=2)

    def count(m):
        return sum(p.numel() for p in m.parameters())

    print(f"输入: {tuple(x.shape)}\n")
    print(f"{'模块':<12}{'中间维度':>10}{'参数量':>12}{'输出形状':>16}")
    print("-" * 52)
    for name, m in (("ReLUFFN", relu_ffn), ("SwiGLUFFN", swiglu_ffn), ("MoEFFN", moe_ffn)):
        y = m(x)
        mid = getattr(m, "expert_hidden", None) or getattr(m, "intermediate_size", 4 * D)
        print(f"{name:<12}{mid:>10}{count(m):>12,}{str(tuple(y.shape)):>16}")

    ratio = count(swiglu_ffn) / count(relu_ffn)
    print(f"\n参数量比值 SwiGLU / ReLU = {ratio:.3f}")
    print("对齐到 256 的整数倍会让小模型的比值偏大；D 越大越接近 1.0。")

    # ---- MoE 的稀疏性 ----
    print("\n--- MoEFFN 稀疏激活 ---")
    total = count(moe_ffn)
    per_expert = count(moe_ffn.experts[0])
    active = count(moe_ffn.router) + per_expert * moe_ffn.top_k
    print(f"专家数 {moe_ffn.num_experts}，每 token 激活 {moe_ffn.top_k} 个")
    print(f"总参数量     : {total:,}")
    print(f"每 token 激活: {active:,}  ({active / total:.1%})")

    # 路由是否均匀：统计每个专家被选中的次数
    with torch.no_grad():
        flat = x.reshape(-1, D)
        probs = F.softmax(moe_ffn.router(flat), dim=-1)
        _, top_i = torch.topk(probs, moe_ffn.top_k, dim=-1)
        load = torch.bincount(top_i.reshape(-1), minlength=moe_ffn.num_experts)
    print(f"专家负载分布 : {load.tolist()}  （未训练，接近随机是正常的）")
    print(f"负载均衡损失 : {moe_ffn.aux_loss.item():.4f}  （均匀约 1.0，越大越不均衡）")
