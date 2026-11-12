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


# Shared-Expert MoE FFN：在 MoE 的基础上加「共享专家」
#
# DeepSeekMoE（DeepSeek-V2 / V3）的做法：
#   1. 一部分专家是「共享专家」(shared expert)，每个 token 都必过，不经过 router
#   2. 另一部分是「路由专家」(routed expert)，仍然由 router 选 top-k
#
# 为什么要这么做：
#   - 共享专家负责承载「所有 token 都需要」的公共知识（语法、常见搭配）
#   - 路由专家只需要学「差异化」的部分，不必每个专家都重复学一遍公共知识
#   - 结果是在同样的总参数量下，专家的「专业化程度」更高，冗余更少
#
# 与 MoEFFN 的区别：
#   1. 多一个 shared_experts 模块，不走 router，所有 token 无条件通过
#   2. 负载均衡损失只统计路由专家（共享专家本来就人人有份，不需要均衡）
#   3. 每 token 激活参数 = 共享专家 + top_k 个路由专家，比纯 MoE 多一份固定开销
class SharedExpertMoEFFN(nn.Module):
    def __init__(self, hidden_size, num_experts=8, top_k=2, num_shared=1,
                 expert_hidden=None, shared_hidden=None,
                 multiple_of=256, bias=False):
        super().__init__()

        if top_k > num_experts:
            raise ValueError(f"top_k({top_k}) 不能大于路由专家数({num_experts})")

        self.num_experts = num_experts
        self.top_k = top_k
        self.num_shared = num_shared

        # 路由网络：只在路由专家里选，共享专家不参与
        self.router = nn.Linear(hidden_size, num_experts, bias=False)

        # 路由专家：被 top-k 选中的 token 才会进来
        self.experts = nn.ModuleList(
            [SwiGLUFFN(hidden_size, expert_hidden, multiple_of, bias)
             for _ in range(num_experts)]
        )
        self.expert_hidden = self.experts[0].intermediate_size

        # 共享专家：所有 token 无条件通过，相当于一个「人人有份」的稠密分支
        self.shared_experts = nn.ModuleList(
            [SwiGLUFFN(hidden_size, shared_hidden, multiple_of, bias)
             for _ in range(num_shared)]
        )
        self.shared_hidden = self.shared_experts[0].intermediate_size if num_shared else 0

        self.aux_loss = None

    def forward(self, x):
        # x: [Batch, Seq, D]
        B, T, D = x.shape
        flat = x.reshape(B * T, D)

        # --- 共享分支：所有 token 全走，不做路由 ---
        out = torch.zeros_like(flat)
        for shared in self.shared_experts:
            out = out + shared(flat)

        # --- 路由分支：每个 token 选 top-k 个专家 ---
        logits = self.router(flat)  # [N, E]
        probs = F.softmax(logits, dim=-1)
        top_p, top_i = torch.topk(probs, self.top_k, dim=-1)
        top_p = top_p / top_p.sum(dim=-1, keepdim=True)

        for e, expert in enumerate(self.experts):
            hit = (top_i == e)  # [N, k]
            tok, slot = hit.nonzero(as_tuple=True)
            if tok.numel() == 0:
                continue
            w = top_p[tok, slot]
            y = expert(flat[tok])
            out.index_add_(0, tok, y * w.unsqueeze(-1))

        # 负载均衡只约束路由专家
        self.aux_loss = self._aux_loss(probs, top_i)
        return out.view(B, T, D)

    def _aux_loss(self, probs, top_i):
        """Switch Transformer 的负载均衡损失，只统计路由专家。"""
        E = self.num_experts
        counts = torch.zeros(E, device=probs.device)
        ones = torch.ones(top_i.numel(), device=probs.device)
        counts.index_add_(0, top_i.reshape(-1), ones)

        f = counts / counts.sum()
        P = probs.mean(dim=0)
        return E * (f * P).sum()


if __name__ == "__main__":
    torch.manual_seed(0)

    B, T, D = 2, 8, 512
    x = torch.randn(B, T, D)

    relu_ffn = ReLUFFN(D, 4 * D)
    swiglu_ffn = SwiGLUFFN(D)
    moe_ffn = MoEFFN(D, num_experts=8, top_k=2)
    shared_moe = SharedExpertMoEFFN(D, num_experts=8, top_k=2, num_shared=1)

    def count(m):
        return sum(p.numel() for p in m.parameters())

    print(f"输入: {tuple(x.shape)}\n")
    print(f"{'模块':<20}{'中间维度':>10}{'参数量':>12}{'输出形状':>16}")
    print("-" * 60)
    for name, m in (("ReLUFFN", relu_ffn), ("SwiGLUFFN", swiglu_ffn),
                    ("MoEFFN", moe_ffn), ("SharedExpertMoEFFN", shared_moe)):
        y = m(x)
        mid = getattr(m, "expert_hidden", None) or getattr(m, "intermediate_size", 4 * D)
        print(f"{name:<20}{mid:>10}{count(m):>12,}{str(tuple(y.shape)):>16}")

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

    # ---- 共享专家：多一份恒定开销，换来更干净的分工 ----
    print("\n--- SharedExpertMoEFFN vs MoEFFN ---")
    total_s = count(shared_moe)
    shared_p = sum(count(m) for m in shared_moe.shared_experts)
    per_expert_s = count(shared_moe.experts[0])
    active_s = count(shared_moe.router) + per_expert_s * shared_moe.top_k + shared_p
    print(f"路由专家 {shared_moe.num_experts} 个（每 token 取 {shared_moe.top_k}）"
          f" + 共享专家 {shared_moe.num_shared} 个（人人必过）")
    print(f"总参数量     : {total_s:,}  （纯 MoE 为 {total:,}，多出 {total_s - total:,}）")
    print(f"每 token 激活: {active_s:,}  ({active_s / total_s:.1%})")
    print(f"其中共享部分 : {shared_p:,}  ({shared_p / active_s:.1%} 的激活参数是共享的)")
    print("共享专家的参数人人都用，所以激活占比会明显高于纯 MoE；"
          "换来的是路由专家不必重复学公共知识。")

    with torch.no_grad():
        flat = x.reshape(-1, D)
        probs = F.softmax(shared_moe.router(flat), dim=-1)
        _, top_i = torch.topk(probs, shared_moe.top_k, dim=-1)
        load = torch.bincount(top_i.reshape(-1), minlength=shared_moe.num_experts)
    print(f"路由负载分布 : {load.tolist()}")
    print(f"负载均衡损失 : {shared_moe.aux_loss.item():.4f}  （只统计路由专家）")
