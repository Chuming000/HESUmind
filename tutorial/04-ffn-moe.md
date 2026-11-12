# 04 · MoE FFN：用稀疏激活换模型容量

> 对应代码：[`FFN/ffn.py`](../FFN/ffn.py) 中的 `MoEFFN`
>
> 一句话：把一个 FFN 拆成 N 个「专家」，每个 token 只走其中 top-k 个。参数量按 N 倍涨，计算量只按 k 倍涨。

---

## 0. 本节要点

- MoE 的核心不是「多个模型」，而是**把参数量和计算量解耦**。
- router 给每个 token 打分，取 top-k，权重要**重归一化**。
- 稀疏必须真稀疏：只把命中的 token 送进专家，而不是算完再乘 mask。
- 没有负载均衡损失，训练后期会塌缩到少数几个专家上。

---

## 1. 发展历程

| 年份 | 工作 | 关键点 |
| :-- | :-- | :-- |
| 1991 | Adaptive Mixture of Local Experts | 最早的 MoE 思想，用于小规模分类 |
| 2017 | Outrageously Large Neural Networks（[arXiv:1701.06538](https://arxiv.org/abs/1701.06538)） | 首次把 MoE 做到千亿参数，提出 top-k 门控与负载均衡 |
| 2020 | GShard（[arXiv:2006.16668](https://arxiv.org/abs/2006.16668)） | 把 MoE 和专家并行（expert parallelism）结合，训练 600B 模型 |
| 2021 | Switch Transformer（[arXiv:2101.03961](https://arxiv.org/abs/2101.03961)） | 简化成 top-1，提出 `E·Σ(f_i·P_i)` 负载均衡损失，训到 1.6T |
| 2022 | ST-MoE（[arXiv:2202.08906](https://arxiv.org/abs/2202.08906)） | 系统研究稳定性、router z-loss、容量因子 |
| 2023 | Mixtral 8×7B（[arXiv:2401.04088](https://arxiv.org/abs/2401.04088)） | 46.7B 总参数、每 token 激活 12.9B，效果超过 LLaMA-2 70B |
| 2024 | DeepSeekMoE（[arXiv:2401.06066](https://arxiv.org/abs/2401.06066)） | 细粒度专家 + 共享专家，DeepSeek-V3 用到 671B |

**主线**：从「加容量」→「加容量还不加算力」→「加容量不加算力还要能训稳」。每一步都在解决上一代的工程问题：通信、负载均衡、训练稳定性。

---

## 2. 原理

### 2.1 结构

$$
\text{MoE}(x) = \sum_{i \in \text{TopK}(p)} p_i \cdot E_i(x)
$$

其中：

$$
p = \text{softmax}(W_r x), \qquad \text{TopK}(p) = \text{概率最高的 } k \text{ 个专家}
$$

三个组成部分：

| 组件 | 作用 | 参数量 |
| :-- | :-- | :-- |
| router `W_r` | 给每个 token 打 `E` 个分 | `D × E` |
| 专家 `E_i` | 各自是一个完整的 FFN（通常是 SwiGLU） | `E ×` 单个 FFN |
| 组合权重 `p_i` | 决定每个专家贡献多少 | 0 |

### 2.2 稀疏在哪

关键区别：**总参数**和**每 token 计算量**不是一回事。

- 稠密 FFN：参数量 = 计算量（每个 token 都过全部权重）。
- MoE：参数量 $\approx E \times$ 单专家，计算量 $\approx k \times$ 单专家。

Mixtral 8×7B 就是这么来的：8 个专家、每 token 选 2 个，总参数 46.7B，激活 12.9B——**用 12.9B 的算力，拿到接近 46.7B 稠密模型的效果**。

### 2.3 负载均衡损失

如果不管，router 会正反馈：某个专家被选得多 → 梯度多 → 变得更强 → 被选得更多，最后所有 token 挤到一两个专家上，其余专家「饿死」。

Switch Transformer 的辅助损失：

$$
\mathcal{L}_{\text{aux}} = E \cdot \sum_{i=1}^{E} f_i \cdot P_i
$$

- $f_i$：分给专家 $i$ 的 token 占比（离散的、不可导）
- $P_i$：专家 $i$ 的平均路由概率（连续的、可导）

完全均匀时 $f_i = P_i = 1/E$，辅助损失取到理论最小值 1。全部挤到一个专家时它显著变大。

注意 $f_i$ 不可导，梯度是从 $P_i$ 那一侧回传的——这就是为什么要把「硬分配比例」和「软概率」乘在一起。

---

## 3. 逐行实现

```python
class MoEFFN(nn.Module):
    def __init__(self, hidden_size, num_experts=8, top_k=2, expert_hidden=None,
                 multiple_of=256, bias=False):
        super().__init__()
        if top_k > num_experts:
            raise ValueError(f"top_k({top_k}) 不能大于专家数({num_experts})")

        self.num_experts = num_experts
        self.top_k = top_k
        self.router = nn.Linear(hidden_size, num_experts, bias=False)   # 不带 bias
        self.experts = nn.ModuleList([
            SwiGLUFFN(hidden_size, expert_hidden, multiple_of, bias)
            for _ in range(num_experts)
        ])
        self.expert_hidden = self.experts[0].intermediate_size
        self.aux_loss = None
```

forward 是重点：

```python
    def forward(self, x):
        B, T, D = x.shape
        flat = x.reshape(B * T, D)          # 展平，路由是逐 token 的

        logits = self.router(flat)          # [N, E]
        probs = F.softmax(logits, dim=-1)
        top_p, top_i = torch.topk(probs, self.top_k, dim=-1)   # [N, k]

        # 归一化被选中的权重，使其和为 1
        top_p = top_p / top_p.sum(dim=-1, keepdim=True)

        out = torch.zeros_like(flat)
        for e, expert in enumerate(self.experts):
            hit = (top_i == e)                       # [N, k] 布尔
            tok, slot = hit.nonzero(as_tuple=True)   # 命中专家 e 的 token 下标
            if tok.numel() == 0:
                continue                             # 没人选，直接跳过

            w = top_p[tok, slot]                     # 每个命中 token 的权重
            y = expert(flat[tok])                    # 只算命中的 token
            out.index_add_(0, tok, y * w.unsqueeze(-1))
        ...
```

**为什么不用「全算一遍再乘 mask」？**
因为那就失去意义了：所有专家都前向一次，计算量等于 $E$ 倍稠密 FFN，MoE 唯一的优势（省算力）荡然无存。`gather → 专家 → scatter` 才是真稀疏。

**`index_add_` 为什么能处理重复下标？**
一个 token 可能同时选中同一个专家吗？不会（top-k 选的是不同专家），但同一个专家会被**多个 token** 选中，那些 token 的下标各不相同。`index_add_` 按 `tok` 累加，正好把「散回原位置」这件事做对。如果手写 `out[tok] += ...`，遇到重复下标会静默出错。

**`w.unsqueeze(-1)` 是为了广播**：`w` 是 `[M]`，`y` 是 `[M, D]`，变成 `[M, 1]` 才能逐行缩放。

负载均衡损失：

```python
    def _aux_loss(self, probs, top_i):
        E = self.num_experts
        counts = torch.zeros(E, device=probs.device)
        ones = torch.ones(top_i.numel(), device=probs.device)
        counts.index_add_(0, top_i.reshape(-1), ones)   # 每个专家被选中多少次

        f = counts / counts.sum()      # 实际分配比例
        P = probs.mean(dim=0)          # 平均路由概率
        return E * (f * P).sum()
```

注意 `top_i.numel() = N * k`——统计的是「槽位」而不是「token」，所以 16 个 token、top-2 时计数总和是 32。

---

## 4. 跑一遍

```bash
python FFN/ffn.py
```

```text
--- MoEFFN 稀疏激活 ---
专家数 8，每 token 激活 2 个
总参数量     : 18,878,464
每 token 激活: 4,722,688  (25.0%)
专家负载分布 : [3, 7, 3, 4, 2, 2, 6, 5]  （未训练，接近随机是正常的）
负载均衡损失 : 1.0598  （均匀约 1.0，越大越不均衡）
```

逐项核对：

**总参数量**

$$
8 \times 2{,}359{,}296 + 512 \times 8 = 18{,}874{,}368 + 4{,}096 = 18{,}878{,}464
$$

（8 个 SwiGLU 专家的参数 + router 的 $D \times E$）

**每 token 激活量**

$$
4{,}096 + 2 \times 2{,}359{,}296 = 4{,}722{,}688
$$

（router + top-2 个专家）

**比例**： $4{,}722{,}688 / 18{,}878{,}464 = 25.0\%$，正好是 $k/E = 2/8$。

**负载分布** `[3,7,3,4,2,2,6,5]`：总和 32 = 16 token × top-2 ✓。最大值 7、最小值 2，还没训练，这种偏差是正常的随机波动，对应的 aux loss 是 1.0598，只比理论最小值 1.0 高一点。

---

## 5. 手算核对：Mixtral 的 46.7B / 12.9B

Mixtral 8×7B 的公开数据：**总参数 46.7B，每 token 激活 12.9B**。按「共享部分 + 8 个专家」拆开看：

| 部分 | 参数量（约） | 说明 |
| :-- | --: | :-- |
| 共享部分（embedding + attention + norm） | 1.9B | 所有 token 都要过 |
| 单个专家（FFN） | 5.6B | 只有被选中的 token 才过 |
| 总计 | `1.9 + 8×5.6 ≈ 46.7` | 对得上 |
| 每 token 激活 | `1.9 + 2×5.6 ≈ 13.1` | 官方报 12.9，取整口径差异 |

所以「8×7B」这个命名指的是**单个专家的量级约 7B**，不是「8 个 7B 模型拼起来 56B」——注意力等部分是共享的，这也是为什么总参数只有 46.7B。

核心比例关系：激活量 / 总参数 $\approx 1.9/46.7 \approx 27\%$，和本仓库 `top_k/num_experts = 2/8 = 25%` 是同一个逻辑（差异来自共享部分不参与路由）。不同模型的专家粒度和共享比例不同，数字会有出入，但「总参数 ≫ 激活参数」这个结构不变。

---

## 6. 易错点

| # | 坑 | 现象 | 正确做法 |
| :-- | :-- | :-- | :-- |
| 1 | top-k 权重不归一化 | 输出整体偏小，训练不稳定 | `top_p / top_p.sum(...)` |
| 2 | 用「全专家计算 + mask」实现 | 显存和算力都是 `E` 倍，MoE 白做 | `gather → expert → index_add_` |
| 3 | 忘了跳过空专家 | 空张量送进 Linear 报错，或白算一次 | `if tok.numel() == 0: continue` |
| 4 | 手写 `out[tok] += y` | 重复下标时静默出错（只保留最后一个） | 用 `index_add_` |
| 5 | 不加 aux loss | 训练几百步后所有 token 挤到 1~2 个专家 | 加 `E·Σ(f_i·P_i)`，系数约 0.01 |
| 6 | aux loss 里用 token 数而非槽位数 | 数值偏大 `k` 倍，和论文对不上 | 统计 `top_i.reshape(-1)` |
| 7 | 训练时随机路由、推理时贪心 | 两边分布不一致，评估掉点 | 保持同一套 top-k 逻辑 |
| 8 | `expert_hidden` 按 4D 给 | 参数量是稠密 FFN 的 1.5 倍，白多花显存 | MoE 专家通常沿用 SwiGLU 的 8/3D |
| 9 | 分布式下各卡专家不同却全量通信 | 带宽打满，吞吐骤降 | 专家并行 + all-to-all，只传命中 token |
| 10 | 没有 capacity factor 概念 | 真实训练时某专家爆满，被迫丢 token | 设 `capacity_factor`（如 1.25），超出丢弃并记录丢弃率 |

第 5 条是 MoE 训练最常见的崩法，而且**指标上看不出来**：loss 还在降，但模型容量已经被浪费在少数专家上。诊断方法就是定期打印负载分布，看它是不是越来越偏。

---

## 7. 大厂高频面试题

**Q1：MoE 到底省了什么？**

省的是**计算量**，不是显存。总参数（权重）还是全都在显存里，只是每个 token 只激活其中一部分。所以 MoE 适合「想要更大容量、但不想让每 token 的 FLOPs 线性增长」的场景；显存不够的问题它解决不了，反而更严重。

**Q2：为什么需要负载均衡损失？不加会怎样？**

router 的梯度会强化「已经受欢迎的专家」，形成正反馈，最终少数专家吃掉几乎所有 token，其余专家参数几乎不更新（dead experts）。辅助损失通过惩罚「分配比例不均」来对抗这个趋势。

**Q3：top-k 的权重为什么要重归一化？**

`softmax` 之后所有专家的概率和为 1，但只取了 top-k，这 k 个的和小于 1。不归一化的话，输出幅度会随 k 变化而缩小，和残差相加时量级不匹配。归一化后 $\sum p_i = 1$，输出的尺度才稳定。

**Q4：MoE 训练为什么容易不稳定？**

三个来源：① router 的 softmax 在低精度下容易饱和，梯度爆炸（ST-MoE 用 router z-loss 缓解）；② 负载不均衡导致部分专家梯度方差极大；③ 专家并行的 all-to-all 通信有延迟，和计算重叠不好会拖慢训练。工程上还要处理 token drop、capacity factor、专家初始化等问题。

**Q5：capacity factor 是什么？**

每个专家能接收的 token 数上限 = `capacity_factor × (总 token 数 / 专家数)`。超出上限的 token 会被丢弃（或走残差直通），用来保证每个专家张量形状固定、便于并行。设太小丢 token 多，设太大浪费显存，常用 1.0~1.5。

**Q6：Mixtral 8×7B 为什么叫 8×7B，实际只有 46.7B？**

因为「7B」指的是单个专家的规模量级，不是「8 个独立的 7B 模型」。注意力层、embedding 等是共享的，8 个专家只替换了 FFN 部分，所以总参数远小于 56B。

**Q7：怎么诊断专家塌缩？**

打印每个专家被选中的 token 比例和平均路由概率。健康状态下各专家占比接近 $1/E$；如果某个专家长期低于 1% 而另一个超过 50%，就是塌缩。也可以看 aux loss 曲线，长期高于 1.2~1.5 就说明分配很不均。

**Q8：共享专家（shared expert）是干什么的？**

DeepSeekMoE 提出：留 1~2 个专家**每个 token 都走**，用来承载通用知识；其余专家做稀疏路由，负责专门化。这样能减少专家之间的知识冗余，让稀疏专家更专注。

---

## 8. 相关论文

| 论文 | 链接 | 为什么值得读 |
| :-- | :-- | :-- |
| Outrageously Large Neural Networks | [arXiv:1701.06538](https://arxiv.org/abs/1701.06538) | MoE 用于 NLP 的开山之作 |
| Switch Transformers | [arXiv:2101.03961](https://arxiv.org/abs/2101.03961) | top-1 路由 + 负载均衡损失的出处 |
| GShard | [arXiv:2006.16668](https://arxiv.org/abs/2006.16668) | 专家并行的工程细节 |
| ST-MoE | [arXiv:2202.08906](https://arxiv.org/abs/2202.08906) | 训练稳定性、z-loss、capacity factor |
| Mixtral of Experts | [arXiv:2401.04088](https://arxiv.org/abs/2401.04088) | 8×7B 的公开配置与效果 |
| DeepSeekMoE | [arXiv:2401.06066](https://arxiv.org/abs/2401.06066) | 细粒度专家 + 共享专家 |

---

## 9. 小结 & 思考题

**小结**：MoE 把 FFN 拆成 N 个专家，每个 token 只走 top-k 个，用稀疏激活换模型容量。三个必要条件：真稀疏的实现、重归一化的 top-k 权重、防塌缩的负载均衡损失。

**思考题**：

1. `top_k = 1` 和 `top_k = num_experts` 分别退化成了什么？后者为什么没有意义？
2. 如果把 aux loss 的系数调成 0，训练 1000 步后负载分布大概会变成什么样？调成 1.0 呢？
3. 为什么 `index_add_` 能正确处理「多个 token 选中同一专家」，而 `out[tok] = y` 不行？
4. 假设 `E=8, k=2`，但所有 token 都均匀分布，aux loss 的理论最小值是多少？用代码验证一下。
5. 专家并行的通信量怎么估？如果 `top_k` 从 2 涨到 4，all-to-all 的数据量翻几倍？

---

## 10. 进阶：带共享专家的 MoE（DeepSeekMoE）

前面的 `MoEFFN` 有个隐含假设：**所有知识都该由路由专家学**。但现实里，token 需要的知识分两类——

- **公共知识**：语法、常见搭配、通用推理套路，几乎每个 token 都要用
- **专门知识**：数学、代码、多语言、领域术语，只有部分 token 用得上

纯路由 MoE 的问题在于：为了让每个专家都能处理常见 token，**每个专家都得把公共知识学一遍**，这是巨大的冗余。专家的参数量被浪费在重复学习上，专业化程度自然上不去。

DeepSeekMoE（DeepSeek-V2 / V3 采用）的解法很直接：**拆出「共享专家」**。

```
                ┌──────────────┐
   x ──────────►│  共享专家      │──────────┐
     │          │ (人人必过)     │          │
     │          └──────────────┘          │
     │                                    ▼
     │          ┌──────────────┐        ┌─────┐
     └─router─►│ 路由专家 top-k │───────►│  +  │──► y
                └──────────────┘        └─────┘
```

- **共享专家**：不经过 router，每个 token 无条件通过，负责公共知识
- **路由专家**：仍然 top-k，但只需要学「共享专家没覆盖到的差异化部分」

这样在总参数量相同时，路由专家的分工更干净；反过来说，达到同样效果所需的专家数更少。

### 实现

`FFN/ffn.py` 里的 `SharedExpertMoEFFN` 是在 `MoEFFN` 上加了 `shared_experts`：

```python
# 共享分支：所有 token 全走，不做路由
out = torch.zeros_like(flat)
for shared in self.shared_experts:
    out = out + shared(flat)

# 路由分支：每个 token 选 top-k 个专家（与 MoEFFN 完全一致）
logits = self.router(flat)
probs = F.softmax(logits, dim=-1)
top_p, top_i = torch.topk(probs, self.top_k, dim=-1)
...
```

两个细节值得注意：

1. **负载均衡只约束路由专家**。共享专家本来就人人有份，不存在「塌缩」问题，把它算进 aux loss 反而会干扰。
2. **共享专家的开销是恒定的**。不管 top_k 是几，每个 token 都要过一遍共享专家，所以激活参数里它的占比是固定的。

### 三种结构的对比（D=512，8 个路由专家，top-2）

| 结构 | 总参数量 | 每 token 激活 | 激活占比 |
| --- | --- | --- | --- |
| MoEFFN（纯路由） | 18,878,464 | 4,722,688 | 25.0% |
| SharedExpertMoEFFN（+1 共享） | 21,237,760 | 7,081,984 | 33.3% |

多出来的 2,359,296 参数就是那个共享专家（一个完整的 SwiGLU FFN）。激活占比从 25% 涨到 33.3%，因为共享部分**人人都在用**，这钱花得值。

### 什么时候该用

- 专家数多、领域跨度大（数学 + 代码 + 多语言）时收益明显
- 小模型（< 1B）用纯 MoE 往往不如稠密模型，加了共享专家更容易训起来
- 共享专家数量一般取 1，多了就接近稠密模型了，失去稀疏的意义

### 思考题

1. 如果 `num_shared = 0`，`SharedExpertMoEFFN` 退化成什么？
2. 共享专家的中间维度该和路由专家一样大吗？如果只有一半，会损失什么？
3. 为什么 DeepSeek-V3 用 1 个共享专家 + 256 个路由专家，而不是反过来？
