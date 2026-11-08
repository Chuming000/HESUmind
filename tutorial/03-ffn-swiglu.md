# 03 · SwiGLU FFN：给前馈网络加一扇门

> 对应代码：[`FFN/ffn.py`](../FFN/ffn.py) 中的 `SwiGLUFFN`
>
> 一句话：多一条门控分支，用 `SiLU` 算出的门去逐元素缩放内容分支。LLaMA / PaLM / Qwen 都用它。

---

## 0. 本节要点

- GLU 家族的核心是**乘法门控**：`(门 ⊙ 内容)`，门由输入动态算出。
- SwiGLU 的激活只作用在 **gate** 分支，`up` 分支不激活——这是最容易写错的地方。
- 中间维度从 4D 缩到 **8/3 D**，再**向上对齐到 256 的倍数**，是为了让参数量与 ReLU FFN 持平。
- LLaMA-2 的 `D=4096 → M=11008` 就是这么算出来的。

---

## 1. 发展历程

| 年份 | 工作 | 关键点 |
| :-- | :-- | :-- |
| 2017 | Language Modeling with Gated Convolutional Networks（[arXiv:1612.08083](https://arxiv.org/abs/1612.08083)） | 提出 **GLU**：`(X*W+b) ⊗ σ(X*V+c)`，门控第一次用于语言模型 |
| 2020 | GLU Variants Improve Transformer（[arXiv:2002.05202](https://arxiv.org/abs/2002.05202)） | 系统比较 ReLU / GeLU / SwiGLU / GeGLU，**SwiGLU 表现最好** |
| 2022 | PaLM（[arXiv:2204.02311](https://arxiv.org/abs/2204.02311)） | 大规模采用 SwiGLU，中间维度 4D |
| 2023 | LLaMA / LLaMA-2（[arXiv:2302.13971](https://arxiv.org/abs/2302.13971) / [2307.09288](https://arxiv.org/abs/2307.09288)） | SwiGLU + 8/3 D + 对齐 256，成为开源标准 |
| 2023+ | Qwen、Mistral、DeepSeek | 全部沿用 SwiGLU 结构 |

**主线**：从「逐元素激活」到「门控激活」。ReLU 只看自己这一个数，SwiGLU 让每个通道的放行比例**由当前 token 的输入决定**——多了一层「条件计算」的意味。

---

## 2. 原理

### 2.1 先看 GLU

$$
\text{GLU}(x) = (W_1 x + b_1) \otimes \sigma(W_2 x + b_2)
$$

$\otimes$ 是逐元素乘，$\sigma$ 是 sigmoid。一条分支当「内容」，另一条当「门」，门的取值在 $(0,1)$，等于给每个通道配了一个软开关。

### 2.2 SwiGLU

把 sigmoid 换成 SiLU（也叫 Swish）：

$$
\text{SiLU}(z) = z \cdot \sigma(z)
$$

于是：

$$
\text{SwiGLU}(x) = \big(\text{SiLU}(W_{\text{gate}} x)\big) \otimes (W_{\text{up}} x)
$$

$$
\text{FFN}(x) = W_{\text{down}} \cdot \text{SwiGLU}(x)
$$

三条投影：`gate`、`up`、`down`。注意：

- **只有 gate 分支过激活**，up 分支原样参与相乘；
- 没有 bias（LLaMA 系的做法）；
- 门可以是负数（SiLU 在 $z<0$ 时输出小负数），比 sigmoid 的 $(0,1)$ 更灵活。

### 2.3 为什么是 8/3 D

参数量的账：

$$
\text{ReLU FFN}: 2DM_{\text{relu}} = 2D \cdot 4D = 8D^2
$$

$$
\text{SwiGLU}: 3DM_{\text{swi}} = 3D \cdot M_{\text{swi}}
$$

让两者相等：

$$
3D M_{\text{swi}} = 8D^2 \;\Longrightarrow\; M_{\text{swi}} = \frac{8}{3}D
$$

**所以 8/3 不是拍脑袋，是「三条投影 vs 两条投影」的配平结果。**

对齐到 256 是因为 GPU 上矩阵乘的分块（tensor core 的 tile 大小）和分布式切分都偏好 2 的幂附近的对齐值，不对齐会有 padding 浪费。

---

## 3. 逐行实现

```python
class SwiGLUFFN(nn.Module):
    def __init__(self, hidden_size, intermediate_size=None, multiple_of=256, bias=False):
        super().__init__()

        if intermediate_size is None:
            # 三条投影 → 中间维度缩到 8/3 D，再向上对齐
            intermediate_size = int(8 * hidden_size / 3)
            intermediate_size = multiple_of * ((intermediate_size + multiple_of - 1) // multiple_of)

        self.intermediate_size = intermediate_size

        self.w_gate = nn.Linear(hidden_size, intermediate_size, bias=bias)   # 门控分支
        self.w_up   = nn.Linear(hidden_size, intermediate_size, bias=bias)   # 内容分支
        self.w_down = nn.Linear(intermediate_size, hidden_size, bias=bias)   # 输出投影

    def forward(self, x):
        gate = F.silu(self.w_gate(x))   # 只有这里过激活
        up   = self.w_up(x)             # 不激活
        x = gate * up                   # 逐元素相乘
        x = self.w_down(x)
        return x
```

三个细节：

**① `int()` 是截断，不是四舍五入**
`int(8*512/3)` = `int(1365.33)` = 1365。随后向上对齐到 256：`256 * ceil(1365/256) = 256 * 6 = 1536`。用 `(a + m - 1) // m` 是整数向上取整的惯用写法（避免 `math.ceil` 引入浮点误差）。

**② `bias=False` 是默认值**
LLaMA 系不用 bias。如果你从 ReLU FFN 抄代码过来，忘了改，会多出 `2M + D` 个参数——数量不大，但和论文里的参数表对不上。

**③ `gate * up` 的顺序无所谓，但语义要清楚**
乘法可交换，但读代码的人需要知道谁是门、谁是内容。养成 `gate * up` 的写法，和论文一致。

---

## 4. 跑一遍

```bash
python FFN/ffn.py
```

```text
模块                中间维度         参数量            输出形状
----------------------------------------------------
ReLUFFN           2048   2,099,712     (2, 8, 512)
SwiGLUFFN         1536   2,359,296     (2, 8, 512)
MoEFFN            1536  18,878,464     (2, 8, 512)

参数量比值 SwiGLU / ReLU = 1.124
对齐到 256 的整数倍会让小模型的比值偏大；D 越大越接近 1.0。
```

**为什么比值是 1.124 而不是 1.0？**

手算：

$$
\text{理想 } M = \frac{8}{3} \times 512 = 1365.33
$$

但要对齐到 256 → **1536**，比理想值多了 12.5%。于是：

$$
\frac{3 \times 512 \times 1536}{2 \times 512 \times 2048} = \frac{2{,}359{,}296}{2{,}097{,}152} = 1.124
$$

**注意分母没有 bias**（SwiGLU 是 `bias=False`，ReLU 是 `bias=True`），严格比较时要把 ReLU 的 2,560 个 bias 参数加回去，比值会略低一点点。

---

## 5. 手算核对：LLaMA-2 的 11008 从哪来

LLaMA-2 7B：`hidden_size = 4096`。

$$
\frac{8}{3} \times 4096 = 10922.67 \;\xrightarrow{\text{int}}\; 10922
$$

$$
256 \times \left\lceil \frac{10922}{256} \right\rceil = 256 \times 43 = 11008
$$

和官方配置里的 `intermediate_size: 11008` 完全一致。

再核对参数量：

| 方案 | 计算 | 参数量 |
| :-- | :-- | --: |
| ReLU FFN（4D） | $2 \times 4096 \times 16384$ | 134,217,728 |
| SwiGLU（不对齐） | $3 \times 4096 \times 10922$ | 134,213,632 |
| SwiGLU（对齐 11008） | $3 \times 4096 \times 11008$ | 135,266,304 |

不对齐时两者差 0.003%，几乎完全相等——这就是 8/3 的用途。对齐后多 0.78%，可以接受。

可以看出：**D 越大，对齐带来的相对开销越小**。`D=512` 时是 12.4%，`D=4096` 时只有 0.78%。这就是为什么小模型跑出来比值偏差明显，别以为自己写错了。

---

## 6. 易错点

| # | 坑 | 现象 | 正确做法 |
| :-- | :-- | :-- | :-- |
| 1 | 两条分支都过激活 | 表达能力下降，参数量对得上但效果变差 | **只有 gate 过 `F.silu`** |
| 2 | 用 sigmoid 代替 SiLU | 门被限制在 (0,1)，负半轴完全截断 | `F.silu(z) = z * sigmoid(z)` |
| 3 | 忘记向上对齐 256 | 和 LLaMA 官方配置的 11008 对不上 | `multiple_of * ceil(M / multiple_of)` |
| 4 | 用 `round` 而不是向上取整 | 4096 时会算出 10752，比官方小 | 必须 `ceil` |
| 5 | 保留 `bias=True` | 参数量比论文多 `2M + D` | `bias=False` |
| 6 | 把 `w_down` 写成 `w_up` 的逆操作 | 维度对不上，直接报错 | 三个投影维度：`D→M`、`D→M`、`M→D` |
| 7 | 以为 8/3 是「为了更好」 | 说不出理由 | 8/3 是**参数配平**的结果，不是效果调优 |
| 8 | 用 `int()` 向下取整后忘记对齐 | 小模型上参数量比 ReLU 少一截 | 先 `int` 再 `ceil` 对齐 |

第 1 条最致命：代码能跑、形状对、loss 也在降，但你已经把它退化成一个「带多余分支的 GeLU FFN」了。

---

## 7. 大厂高频面试题

**Q1：SwiGLU 相比 ReLU FFN 改了什么？**

两处：① 把单条「升维→激活→降维」改成「两条升维分支，一条做门、一条做内容，相乘后降维」；② 中间维度从 4D 缩到 8/3 D，保持总参数量不变。本质是把「逐元素激活」升级成「条件门控」。

**Q2：为什么中间维度是 8/3 D？**

因为多了一条投影。参数量要配平：ReLU 是 $2D \cdot 4D = 8D^2$，SwiGLU 是 $3D \cdot M$，令相等得 $M = 8D/3$。再由工程原因向上对齐到 256 的倍数，LLaMA-2 就是 $4096 \to 11008$。

**Q3：门控为什么能提升效果？**

门由输入动态计算，等于给每个通道一个「随 token 变化」的缩放系数。不同 token 的同一个通道可以被放大或抑制，模型因此获得了**条件计算**能力。论文在相同算力下对比，带门控的变体困惑度更低。

**Q4：SwiGLU 和 GeGLU 有什么区别？**

结构一样，只是激活函数不同：GeGLU 用 GeLU 做门，SwiGLU 用 SiLU。Shazeer 的论文里 SwiGLU 略优，且 SiLU 计算更便宜（一个 sigmoid 乘法），所以 LLaMA 选了 SwiGLU。

**Q5：为什么 SwiGLU 常常不带 bias？**

原始 GLU 论文和 LLaMA 都不带。原因：门控本身已经提供了乘性的偏移能力，bias 的边际收益很小；去掉能省参数、省一次加法，还能让 norm 的 γ 更容易 fold 进权重。

**Q6：如果显存吃紧，能不能把 SwiGLU 退化成 ReLU FFN？**

可以，但要注意中间维度要改回 4D，参数量才对得上。直接保留 8/3 D 的配置换激活，会让 FFN 变小约 25%，效果下降但省显存——这是一些小模型的实际做法。

---

## 8. 相关论文

| 论文 | 链接 | 为什么值得读 |
| :-- | :-- | :-- |
| GLU Variants Improve Transformer | [arXiv:2002.05202](https://arxiv.org/abs/2002.05202) | SwiGLU 的出处，含各变体对比表 |
| Language Modeling with Gated Convolutional Networks | [arXiv:1612.08083](https://arxiv.org/abs/1612.08083) | GLU 的原始定义 |
| PaLM | [arXiv:2204.02311](https://arxiv.org/abs/2204.02311) | 大规模 SwiGLU 的工程实践 |
| LLaMA-2 | [arXiv:2307.09288](https://arxiv.org/abs/2307.09288) | 11008 这个数字的来源 |

---

## 9. 小结 & 思考题

**小结**：SwiGLU = 门控 + SiLU，三条投影，中间维度 8/3 D 且对齐 256。它用「多一个矩阵」换「更强的条件表达」，同时通过缩维度把参数量摁回和 ReLU 持平。

**思考题**：

1. 如果 `multiple_of` 设成 1，`D=512` 时 SwiGLU 的参数量是多少？和 ReLU 差多少？
2. 为什么 `int()` 之后要**向上**对齐，而不是向下？向下会损失什么？
3. 门分支用 `SiLU`、内容分支不激活——如果把两者调换（内容过 SiLU、门不激活），会有什么问题？
4. 推导：当 `D` 趋于无穷时，SwiGLU 与 ReLU FFN 的参数量比值趋于多少？
