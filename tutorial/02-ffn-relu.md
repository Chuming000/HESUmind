# 02 · ReLU FFN：最经典的前馈网络

> 对应代码：[`FFN/ffn.py`](../FFN/ffn.py) 中的 `ReLUFFN`
>
> 一句话：先升维、再过激活、再降维。看着简单，但它占了 Transformer 约 2/3 的参数量。

---

## 0. 本节要点

- FFN 是**逐位置**（position-wise）的：同一个线性层独立作用于每个 token，token 之间不交互。
- 升维的作用不是「变复杂」，而是给非线性变换提供**更宽的工作空间**，降维再压回残差流。
- 中间维度取 **4D** 是工程惯例，没有严格理论证明。
- 参数量公式 `2·D·M + D + M`，别漏了 bias。

---

## 1. 发展历程

| 年份 | 工作 | FFN 的形态 |
| :-- | :-- | :-- |
| 2017 | Attention Is All You Need（[arXiv:1706.03762](https://arxiv.org/abs/1706.03762)） | `D → 4D → ReLU → D`，Pre-LN 之外的另一个标配 |
| 2018 | BERT（[arXiv:1810.04805](https://arxiv.org/abs/1810.04805)） | 换成 GeLU，`D=768, M=3072`（还是 4D） |
| 2020 | GPT-3（[arXiv:2005.14165](https://arxiv.org/abs/2005.14165)） | GeLU，`D=12288, M=49152`（还是 4D） |
| 2020 | GLU Variants Improve Transformer（[arXiv:2002.05202](https://arxiv.org/abs/2002.05202)） | 指出 SwiGLU 等在同等算力下更好，**FFN 开始分化** |
| 2023 | LLaMA（[arXiv:2302.13971](https://arxiv.org/abs/2302.13971)） | 换成 SwiGLU，中间维度 8/3 D |
| 2024 | 各类 MoE 模型 | FFN 变成「一堆专家 + 路由器」 |

**主线**：ReLU FFN 是 2017 年的标准答案，统治了五年；2020 年门控变体出现后，2023 年起主流开源模型基本换成 SwiGLU。但**理解 FFN 的结构、参数量、作用，仍然要从 ReLU 版开始**——后面两个都是它的变体。

---

## 2. 原理

### 2.1 公式

$$
\text{FFN}(x) = W_2 \cdot \text{ReLU}(W_1 x + b_1) + b_2
$$

其中 $W_1 \in \mathbb{R}^{M \times D}$， $W_2 \in \mathbb{R}^{D \times M}$， $M$ 是中间维度（通常 $M = 4D$）。

对整个序列 $X \in \mathbb{R}^{B \times T \times D}$ 来说，这个操作作用于**每个位置独立**：

$$
\text{FFN}(X)_{b,t,:} = W_2 \cdot \text{ReLU}(W_1 X_{b,t,:} + b_1) + b_2
$$

实现上就是一个 `nn.Linear`，它天然支持任意前置维度，所以 `[B, T, D]` 进去 `[B, T, M]` 出来，不需要写循环。

### 2.2 它到底在干什么

三个视角，面试都可能问：

**① 非线性变换**
Attention 本身是加权求和，是**线性**的（softmax 是非线性的，但对 value 的组合是线性的）。如果只有 Attention，堆多少层都是线性的。FFN 里的激活函数提供了逐位置的强非线性，模型才有能力拟合复杂函数。

**② key-value memory**
有论文把 FFN 看成一种键值记忆： $W_1$ 的每一行是一个「键」， $W_2$ 的每一列是对应的「值」。输入 $x$ 与所有键做内积，ReLU 做稀疏化（只保留正响应），再加权求和得到输出。这解释了为什么 FFN 参数多——它其实是模型的「知识库」。

**③ 通道混合**
Attention 混合的是 **token 之间**的信息（沿序列维），FFN 混合的是 **通道之间** 的信息（沿特征维）。两者交替，才算完整的 Transformer block。

### 2.3 为什么是 4 倍

没有理论推导，是 2017 年原论文的经验选择，之后所有模型照抄。可能的直觉：

- 升维给非线性提供更宽的空间，降维再把信息压回残差流，类似「先展开再收敛」；
- 参数量上，`M=4D` 时 FFN 参数约为 $8D^2$，Attention 约为 $4D^2$，两者比例约 2:1——一个比较均衡的分配。

---

## 3. 逐行实现

```python
class ReLUFFN(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)      # D -> M
        self.fc2 = nn.Linear(intermediate_size, hidden_size)      # M -> D

    def forward(self, x):
        # x: [Batch, Seq, D]
        x = self.fc1(x)      # [B, T, D] -> [B, T, M]
        x = F.relu(x)        # 非线性，负值置零
        x = self.fc2(x)      # [B, T, M] -> [B, T, D]
        return x
```

没有 bias 的显式声明，因为 `nn.Linear` **默认 `bias=True`**——这一点和后面的 SwiGLU 不一样，注意区分。

参数量：

$$
\underbrace{D \cdot M + M}_{\text{fc1}} + \underbrace{M \cdot D + D}_{\text{fc2}} = 2DM + D + M
$$

代入 $D=512, M=2048$：

$$
2 \times 512 \times 2048 + 512 + 2048 = 2{,}097{,}152 + 2{,}560 = 2{,}099{,}712
$$

---

## 4. 跑一遍

```bash
python FFN/ffn.py
```

`__main__` 里三个 FFN 一起跑，ReLU 那一行：

```text
模块                中间维度         参数量            输出形状
----------------------------------------------------
ReLUFFN           2048   2,099,712     (2, 8, 512)
```

- 中间维度 2048 = `4 × 512`，符合 4D 惯例；
- 参数量 2,099,712，和上面手算完全一致；
- 输出形状回到 `(2, 8, 512)`，残差可以直接相加。

---

## 5. 手算核对

再验一遍「FFN 占 block 参数量的 2/3」这个说法。

一个标准 Transformer block（MHA + FFN），取 $D=512$、4 个头、 $M=4D$：

| 部分 | 参数量公式 | 数值 |
| :-- | :-- | --: |
| Attention（Q/K/V/O 四个 `D×D`） | `4D² + 4D` | 1,050,624 |
| FFN | `2DM + D + M` | 2,099,712 |
| 合计 |  | 3,150,336 |

FFN 占比： $2{,}099{,}712 / 3{,}150{,}336 \approx 66.7\%$。

所以「大模型的参数主要在 FFN 里」不是感觉，是算出来的。这也解释了后面 MoE 为什么专挑 FFN 下手——要加容量，加在 FFN 上最划算。

---

## 6. 易错点

| # | 坑 | 现象 | 正确做法 |
| :-- | :-- | :-- | :-- |
| 1 | 算参数量漏 bias | 手算和 `sum(p.numel())` 对不上，差几千 | 记住 `nn.Linear` 默认带 bias，公式要加 `D + M` |
| 2 | 以为 `nn.Linear` 只能吃二维输入 | 手动 reshape 成 `[B*T, D]` 再变回来，白折腾 | `nn.Linear` 支持任意前置维度 |
| 3 | 把 ReLU 写成 inplace | 和残差/梯度检查点一起用时可能踩到已释放的张量 | 默认 `F.relu(x)`，别随手 `inplace=True` |
| 4 | 忘记激活 | 两个线性层直接串联等价于**一个**线性层，模型退化成线性 | 检查 forward 里有没有激活 |
| 5 | 中间维度随手填 | 显存/参数量和预期差一倍 | 显式写 `4 * hidden_size`，并注释来源 |
| 6 | 以为 FFN 会混合 token | 写成跨 token 的操作，破坏因果性 | FFN 是 position-wise，只在特征维操作 |
| 7 | dropout 位置放错 | 训练/推理行为不一致 | 一般在激活后、第二个线性前（如有需要） |

第 4 条是最隐蔽的：代码能跑、loss 能降，但表达能力被砍掉一大截。写完一定要肉眼确认激活函数在里面。

---

## 7. 大厂高频面试题

**Q1：Transformer 里 FFN 的作用是什么？为什么不能去掉？**

两个层面：① 提供**逐位置的非线性**，Attention 的输出本质是 value 的加权和，没有 FFN 的话整个模型接近线性，表达能力严重不足；② 从记忆视角看，FFN 承担了大部分「知识存储」，参数量占比约 2/3。去掉 FFN，模型会变成一个只能做路由的浅层结构。

**Q2：为什么 FFN 要先升维再降维？直接 `D → D` 不行吗？**

可以，但效果差。升维给非线性变换提供更宽的空间，让不同特征组合有更多自由度；降维再把信息压回残差流需要的维度。等价地说， $M=4D$ 让 FFN 成为一个「宽而浅」的变换，比「窄而深」更容易训练。

**Q3：中间维度为什么是 4 倍？**

工程惯例，源自 2017 年原论文，没有严格理论依据。经验上 4 倍在参数量、显存、效果之间比较平衡。后来的模型会调整这个比例：SwiGLU 用 8/3，MoE 的单个专家通常也用 8/3 或者更小。

**Q4：ReLU、GeLU、SwiGLU 有什么区别？**

ReLU： $\max(0,x)$，计算最省，但负半轴梯度为 0，存在 dead neuron 问题，且在 0 点不可导。
GeLU： $x \cdot \Phi(x)$，用高斯 CDF 做平滑门控，BERT/GPT 系列在用，效果比 ReLU 略好。
SwiGLU：不是单纯的逐元素激活，而是**门控结构**（两条分支相乘），表达能力强于前两者，代价是多一个矩阵。详见[第 03 章](03-ffn-swiglu.md)。

**Q5：FFN 的参数量占整个模型多少？如果显存不够，先砍哪里？**

标准结构下 FFN 约占 2/3（见上面的手算）。显存不够时，FFN 是首选目标：可以缩小中间维度、可以量化 FFN 权重、也可以换成 MoE 把「总参数」和「每 token 计算量」解耦。

**Q6：FFN 能换成卷积或者别的结构吗？**

可以。FFN 本质上是对每个位置做 $1\times1$ 卷积（`kernel_size=1` 的 Conv1d 和 Linear 在数学上等价）。也有工作尝试用卷积或状态空间模型替代，但只要还想保持「逐位置 + 通道混合」， $1\times1$ 卷积就是同一件事的不同写法。

---

## 8. 相关论文

| 论文 | 链接 | 为什么值得读 |
| :-- | :-- | :-- |
| Attention Is All You Need | [arXiv:1706.03762](https://arxiv.org/abs/1706.03762) | FFN 的原始定义与 4D 惯例 |
| BERT | [arXiv:1810.04805](https://arxiv.org/abs/1810.04805) | GeLU 版的 FFN |
| GLU Variants Improve Transformer | [arXiv:2002.05202](https://arxiv.org/abs/2002.05202) | 把 FFN 的各种变体放在同一算力下比较 |
| Transformer Feed-Forward Layers Are Key-Value Memories | [arXiv:2012.14913](https://arxiv.org/abs/2012.14913) | 「FFN 是记忆」这个视角的出处 |

---

## 9. 小结 & 思考题

**小结**：ReLU FFN = 升维 → ReLU → 降维，逐位置、不混 token、占 2/3 参数。它是理解后面所有 FFN 变体的基准线。

**思考题**：

1. 如果去掉 `fc1` 的 bias，参数量变成多少？和 `D=512, M=2048` 的实际数字对一下。
2. 为什么说「两个 Linear 中间不加激活」等价于一个 Linear？写出矩阵形式。
3. FFN 的参数量占比 2/3 是在什么前提下成立的？如果 $M = 2D$，占比变成多少？
4. 如果把 FFN 换成 `1×1` 卷积，输入形状 `[B, D, T]` 和 `[B, T, D]` 分别该怎么写？两者结果一致吗？
