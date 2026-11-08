# 01 · 归一化层：LayerNorm 与 RMSNorm

> 对应代码：[`norm/rms_norm.py`](../norm/rms_norm.py)
>
> 一句话：LayerNorm 做「减均值 + 除标准差」，RMSNorm 把「减均值」这一步砍了，只除均方根——效果基本不掉，速度更快。

---

## 0. 本节要点

- 归一化解决的是**激活值尺度漂移**，让每一层的输入分布稳定，深层网络才训得动。
- LayerNorm 的两个统计量是 **均值** 和 **标准差**；RMSNorm 只用 **均方根（RMS）**。
- 省掉均值中心化后，输出**不再是零均值**——这是特性，不是 bug。
- 手写时最容易翻车的地方不是公式，是 **dtype**：FP16 下 `x²` 会溢出，必须 upcast 到 FP32。

---

## 1. 发展历程

| 年份 | 工作 | 关键点 |
| :-- | :-- | :-- |
| 2015 | Batch Normalization（[arXiv:1502.03167](https://arxiv.org/abs/1502.03167)） | 按 **batch** 维度统计均值方差，CNN 里大杀四方 |
| 2016 | Layer Normalization（[arXiv:1607.06450](https://arxiv.org/abs/1607.06450)） | 改成按 **特征** 维度统计，摆脱对 batch 的依赖，为 RNN/序列模型准备 |
| 2017 | Attention Is All You Need（[arXiv:1706.03762](https://arxiv.org/abs/1706.03762)） | Transformer 采用 **Post-LN**：`x = LN(x + Sublayer(x))` |
| 2019 | Root Mean Square Layer Normalization（[arXiv:1910.07467](https://arxiv.org/abs/1910.07467)） | 去掉均值中心化与 bias，只保留 RMS 缩放 |
| 2020 | On Layer Normalization in the Transformer Architecture（[arXiv:2002.04745](https://arxiv.org/abs/2002.04745)） | 系统论证 **Pre-LN** 更稳，深层 Transformer 不必小心翼翼地 warmup |
| 2023 | LLaMA（[arXiv:2302.13971](https://arxiv.org/abs/2302.13971)） | RMSNorm + Pre-LN 成为开源大模型的事实标准 |

**一句话串起来**：从「按 batch 归一化」→「按特征归一化」→「只缩放不居中」→「放到残差分支外面（Pre-LN）」，每一步都在为**更深的网络、更大的 batch、更省的通信**让路。

---

## 2. 原理

### 2.1 LayerNorm

对一个 token 的隐藏向量 $x \in \mathbb{R}^{d}$：

$$
\mu = \frac{1}{d}\sum_{i=1}^{d} x_i,\qquad
\sigma^2 = \frac{1}{d}\sum_{i=1}^{d}(x_i-\mu)^2
$$

$$
\text{LN}(x) = \gamma \odot \frac{x-\mu}{\sqrt{\sigma^2+\epsilon}} + \beta
$$

其中 $\gamma,\beta \in \mathbb{R}^d$ 是可学习参数， $\epsilon$ 防止除零。

注意方差用的是**有偏估计**（除以 $d$ 而不是 $d-1$），PyTorch 的 `nn.LayerNorm` 也是这么算的。

### 2.2 RMSNorm

$$
\text{RMS}(x) = \sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2},\qquad
\text{RMSNorm}(x) = \gamma \odot \frac{x}{\text{RMS}(x)}
$$

就两处差别：

1. **不减均值**——分母直接用原始向量的均方根；
2. **没有 bias**——只有缩放参数 $\gamma$，没有平移参数 $\beta$。

论文的说法是：LayerNorm 的成功主要来自**再缩放不变性**（re-scaling invariance），均值中心化带来的收益很小，但每次都要多算一遍均值、多一次减法、多一组参数。

省下来的是实打实的：少一次 reduction（分布式训练里还少一次 all-reduce），少一份显存读写。这一点在几十层 × 几千 token 的规模上会被放大。

> ⚠️ 关键认知：RMSNorm 的输出**不是零均值**。如果你后面接了依赖零均值假设的东西（比如某些初始化分析），要自己补中心化。

---

## 3. 逐行实现

```python
class RMSNorm(nn.Module):
    def __init__(self, dim, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))   # γ，初始化为 1
        self.eps = eps

    def _norm(self, x):
        # 先转 FP32：FP16 下 x**2 很容易超过 65504 直接变 inf
        return x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return (self.weight * self._norm(x)).to(x.dtype)   # 结果转回原 dtype
```

几个细节值得单独说：

**① `mean(-1, keepdim=True)` 里的 `-1`**
归一化永远作用在**最后一维**（特征维），不是 batch 维。`keepdim=True` 保证形状是 `[..., 1]`，能和 `[..., d]` 广播。

**② 为什么用 `rsqrt` 而不是 `1 / sqrt`**
`torch.rsqrt` 是单条指令的倒数平方根，比 `1/torch.sqrt(y)` 少一次除法、少一次中间张量。大模型里这类逐元素操作的次数是天文数字。

**③ 为什么 `eps` 加在根号里面**
写法是 $\sqrt{\text{ms}+\epsilon}$ 而不是 $\sqrt{\text{ms}}+\epsilon$。差别在数值上：当 $\text{ms}$ 极小时，前者让分母有下界，后者会让分母趋近 $\epsilon$ 但分子仍可能爆掉。加在里面更稳。

**④ 为什么必须 `.float()`**
FP16 的最大值是 65504。一个隐藏向量里只要出现 $|x| > 256$， $x^2$ 就溢出了，`mean` 得到 `inf`，`rsqrt` 得到 `0`，整个 token 的输出变成 0——而且是静默的，不会报错。转成 FP32 算完再转回去，是 LLaMA 等实现的标准做法。

**⑤ 为什么最后要 `.to(x.dtype)`**
输入是 BF16、中间算 FP32，如果直接返回 FP32，下一层的 Linear 会做一次隐式类型提升，既慢又可能引发 mixed-precision 相关的报错。显式转回去，行为才可预期。

接下来是 LayerNorm，结构几乎对称：

```python
class LayerNorm(nn.Module):
    def __init__(self, dim, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))    # γ
        self.bias = nn.Parameter(torch.zeros(dim))     # β ← RMSNorm 没有这个
        self.eps = eps

    def _norm(self, x):
        x = x.float()
        mean = x.mean(-1, keepdim=True)
        var = (x - mean).pow(2).mean(-1, keepdim=True)  # 有偏方差
        return (x - mean) * torch.rsqrt(var + self.eps)

    def forward(self, x):
        return (self.weight * self._norm(x) + self.bias).to(x.dtype)
```

对比着看，差别一目了然：多一次 `mean`、多一次减法、多一个 `bias` 参数。

---

## 4. 跑一遍

```bash
python norm/rms_norm.py
```

实测输出（`torch 2.13.0+cpu`）：

```text
原始数据：
tensor([[ 1.,  2.,  3.],
        [11., 12., 13.]])

RMSNorm 输出：
tensor([[0.4629, 0.9258, 1.3887],
        [0.9146, 0.9977, 1.0808]])

LayerNorm 输出：
tensor([[-1.2247,  0.0000,  1.2247],
        [-1.2247,  0.0000,  1.2247]])
```

看懂这两组数字，这一节就结束了：

- **LayerNorm 两行结果一模一样**。因为 `[1,2,3]` 和 `[11,12,13]` 只差一个整体平移，而 LayerNorm 第一步就把平移减掉了——这正是它「与整体偏移无关」的体现。
- **RMSNorm 两行完全不同**。第一行 `[0.46, 0.93, 1.39]` 明显小于第二行 `[0.91, 1.00, 1.08]`：第二行整体数值大得多，均方根也大，但**均值没有被减掉**，所以输出仍然偏正、量级不同。

---

## 5. 手算核对

**RMSNorm 第一行**：

$$
\text{RMS} = \sqrt{\frac{1^2+2^2+3^2}{3}} = \sqrt{\frac{14}{3}} = 2.16025
$$

$$
\frac{1}{2.16025} = 0.46291,\quad \frac{2}{2.16025}=0.92582,\quad \frac{3}{2.16025}=1.38873
$$

和输出 `[0.4629, 0.9258, 1.3887]` 完全对上。

**LayerNorm 第一行**：

$$
\mu = 2,\quad \sigma^2 = \frac{(1-2)^2+(2-2)^2+(3-2)^2}{3} = \frac{2}{3},\quad \sigma = 0.81650
$$

$$
\frac{1-2}{0.81650} = -1.2247,\quad 0,\quad 1.2247
$$

对上。顺带一提，`1/0.8165 ≈ 1.2247`——这个数在 LayerNorm 的例子里几乎总是出现，因为三维向量的标准差正好是 $\sqrt{2/3}$。

---

## 6. 易错点

| # | 坑 | 现象 | 正确做法 |
| :-- | :-- | :-- | :-- |
| 1 | 不做 FP32 upcast | FP16/BF16 训练时 loss 突然变 NaN 或大段输出为 0，且**不报错** | 归一化内部统一 `.float()`，出口 `.to(x.dtype)` |
| 2 | `eps` 加在根号外 | 数值小时分母失控，梯度爆炸 | 写成 `rsqrt(ms + eps)` |
| 3 | 方差用了无偏估计 | 和 `nn.LayerNorm` 对不上，验证时以为实现错了 | 分母用 `d`（有偏），PyTorch 也是这么算 |
| 4 | 归一化维度写错 | 对 batch 维做归一化，等价于 BatchNorm，序列任务直接崩 | `dim=-1` |
| 5 | 期望 RMSNorm 输出零均值 | 后续分析的假设不成立 | 记住它只保尺度、不保均值 |
| 6 | 忘记 `weight` 初始化为 1 | 初始输出被随机缩放，训练前期震荡 | `nn.Parameter(torch.ones(dim))` |
| 7 | 在 FP16 下直接用 `x.pow(2).mean()` | 极少数 token 溢出，表现为「偶发」的坏样本 | 同 1，先转 FP32 |

第 1 条和第 7 条是同一个根因，但表现不同：前者是**全程坏**，后者是**偶发坏**。后者更难查，因为大部分 batch 都是好的。

---

## 7. 大厂高频面试题

> 下面这些题在大模型岗面试里反复出现，来源是公开面经与面试题库（如[卡码笔记的大模型专栏](https://notes.kamacoder.com/llm/transformer/modern_llm_improvements.html)）。答的时候别只报名字，要讲清楚**改了哪个瓶颈、机制是什么、代价是什么**。

**Q1：LayerNorm 和 BatchNorm 有什么区别？为什么 NLP 不用 BN？**

BN 在 **batch 维** 统计（同一个通道跨样本），LN 在 **特征维** 统计（同一个样本跨通道）。NLP 不用 BN 的原因：序列长度可变、batch 内样本差异大导致统计量不稳；推理时 batch 可能只有 1，BN 的滑动统计和训练时对不上；LN 与 batch 无关，天然适配变长输入和自回归生成。

**Q2：Pre-LN 和 Post-LN 有什么区别？为什么现在都用 Pre-LN？**

Post-LN（原始 Transformer）是 `x = LN(x + Sublayer(x))`，归一化在残差**之后**；Pre-LN 是 `x = x + Sublayer(LN(x))`，归一化在子层**之前**。Post-LN 的梯度要穿过每一层的 LN，深层容易梯度消失/爆炸，必须配 warmup 才训得动；Pre-LN 让残差路径上是一条「恒等高速公路」，梯度可以直通底层，训练稳定得多。代价是表达能力和最终效果略逊，所以后来又出现了 DeepNorm 这类折中方案。

**Q3：RMSNorm 相比 LayerNorm 少了什么？为什么可以少？**

少了均值中心化和 bias。RMSNorm 论文的出发点是：LayerNorm 的效果主要来自**再缩放不变性**，中心化贡献很小。去掉后仍然对输入的整体缩放不敏感，但计算上少一次均值 reduction、少一次减法、少一组参数。实测精度基本持平。

**Q4：为什么 RMSNorm 里要把输入转成 FP32？**

FP16 的最大值是 65504。隐藏向量里只要出现 $|x|>256$， $x^2$ 就溢出成 `inf`，均值变 `inf`，`rsqrt` 返回 0，整个 token 被静默清零。BF16 虽然动态范围够，但尾数只有 7 位，平方后精度损失明显。所以归一化内部统一用 FP32 算，出口再转回原 dtype。

**Q5：RMSNorm 能不能和相邻的线性层融合？**

能，而且这是它比 LayerNorm 更好优化的原因之一。RMSNorm 只依赖单个 token 自己的向量（不跨 token），且只有缩放没有平移， $\gamma$ 可以直接吸收进后续 Linear 的权重里，推理时省掉一次独立的 norm kernel。这也是为什么很多推理框架会对 Pre-Norm 结构做 fold。

**Q6：为什么 LayerNorm 有 bias，RMSNorm 没有？**

bias 提供的是整体平移能力。但 Pre-LN 结构里，norm 后面紧跟线性层，而线性层本身有 bias，两者的平移是冗余的；再加上 RMSNorm 本身不再中心化，保留 bias 的意义更小。砍掉既省参数也省一次加法。

---

## 8. 相关论文

| 论文 | 链接 | 为什么值得读 |
| :-- | :-- | :-- |
| Layer Normalization | [arXiv:1607.06450](https://arxiv.org/abs/1607.06450) | 原始定义，解释为什么按特征维统计 |
| Root Mean Square Layer Normalization | [arXiv:1910.07467](https://arxiv.org/abs/1910.07467) | RMSNorm 出处，有 7%~64% 的加速数据 |
| On Layer Normalization in the Transformer Architecture | [arXiv:2002.04745](https://arxiv.org/abs/2002.04745) | Pre-LN vs Post-LN 的理论与实验 |
| Attention Is All You Need | [arXiv:1706.03762](https://arxiv.org/abs/1706.03762) | Post-LN 的原始用法 |

---

## 9. 小结 & 思考题

**小结**：归一化是「让每层输入回到可控尺度」的保险丝。LayerNorm 减均值除标准差，RMSNorm 只除均方根。大模型选 RMSNorm，图的是**省一次 reduction、少一组参数、还能和线性层融合**。

**思考题**：

1. 如果输入是一个**全零**向量，RMSNorm 会输出什么？`eps` 在这里起什么作用？
2. 把 `eps` 从 `1e-5` 改成 `1e-8`，在 FP16 下会发生什么？在 FP32 下呢？
3. 为什么 LayerNorm 对 `[1,2,3]` 和 `[11,12,13]` 输出相同，而 RMSNorm 不同？这个差异在什么场景下会变成问题？
4. 如果让你把 RMSNorm 的 `weight` 融进下一层的 `Linear`，公式应该怎么推？（提示：`y = W·(γ⊙x/rms(x))`）
