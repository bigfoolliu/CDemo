# 现代变换器（Transformer）组件对照

本篇从仓库已有的小型 Transformer 出发，认识几种常见组件及其设计目的。它是对照式入门，不表示这些组件已经在项目 C 代码中实现。

## 1. 当前教学模型作为基线

[`network_forward`](../src/ml/12_tiny_transformer.c#L465) 使用可学习的位置向量（learned positional embeddings）、前置层归一化（pre-LayerNorm）、多头因果注意力（multi-head causal attention）、高斯误差线性单元（Gaussian Error Linear Unit，GELU）前馈层和残差连接（residual connection）。完整前向与反向推导见[小型 Transformer 教程](58_小型Transformer.md)。

与此基线对照的组件包括旋转位置编码（rotary positional embedding，RoPE）、均方根归一化（root mean square normalization，RMSNorm）、Swish 门控线性单元（Swish-Gated Linear Unit，SwiGLU）和分组查询注意力（grouped-query attention，GQA）。比较时一次只替换一个组件，否则很难判断变化来自哪里。

## 2. 旋转位置编码：把位置信息放进注意力

学习式位置向量会直接加到词元（token）表示上；旋转位置编码（RoPE）则按位置旋转查询（query）和键（key）的二维特征对，使注意力分数包含位置信息。对位置 $t$、第 $i$ 对特征及其旋转频率 $\omega_i$，查询向量一对分量可写为：

$$
\begin{aligned}
\begin{bmatrix}
q'_{t,2i}\\
q'_{t,2i+1}
\end{bmatrix}
&=
\begin{bmatrix}
\cos(t\omega_i)&-\sin(t\omega_i)\\
\sin(t\omega_i)&\cos(t\omega_i)
\end{bmatrix}
\begin{bmatrix}
q_{t,2i}\\
q_{t,2i+1}
\end{bmatrix}.
\end{aligned}
\tag{1}
$$

$q_{t,2i}$、$q_{t,2i+1}$ 是旋转前的查询分量，带撇号的量是旋转后的结果；键向量也按相同位置规则处理。该变换不增加可学习的位置向量参数，但改变了注意力的数值计算。理解它的第一步是手算一个位置和一对维度，而不是先接入完整模型。

## 3. RMSNorm 与 SwiGLU：归一化和前馈层的替换

RMSNorm 按向量均方根缩放，不减去特征均值；它的表达式和 LayerNorm 的统计范围已在[训练稳定性与归一化](72_训练稳定性与归一化.md)中对照。更换归一化层时，检查输出形状、常量输入和反向梯度，并保持其余网络配置不变。

现有模型的前馈层使用 GELU。SwiGLU 用一个门控分支逐元素调制另一分支：

$$
\begin{aligned}
\operatorname{SwiGLU}(\mathbf{x})
&=\operatorname{SiLU}(\mathbf{x}W_g)
\odot(\mathbf{x}W_v).
\end{aligned}
\tag{2}
$$

这里 $\mathbf{x}$ 是输入特征，$W_g$ 和 $W_v$ 把模型宽度投影到相同的前馈隐藏宽度，$\odot$ 表示逐元素乘法，$\operatorname{SiLU}(u)=u\,\operatorname{sigmoid}(u)$。公式省略了把隐藏结果投影回模型宽度的输出矩阵。该门控通常会改变前馈层参数量；公平比较时应记录参数数目或匹配隐藏宽度，不能只比较训练损失。

## 4. GQA 与 KV 缓存：用缓存内存换重复计算

多头注意力（multi-head attention，MHA）通常为每个查询头各产生一组键和值。GQA 让多个查询头共享一组键和值；例如 $4$ 个查询头、$2$ 个键值头时，每个键值头供两个查询头使用。在头维度和上下文长度相同的条件下，键和值缓存约为对应 MHA 的一半。

键值缓存（key-value cache，KV cache）保存已经处理过的 token 的键和值。生成下一个 token 时，只需计算新位置并读取旧缓存，避免每一步都重新计算整个前缀；这种逐步生成方式称为增量推理（incremental inference），代价是缓存内存随上下文长度增加。GQA 通过减少键值头数降低这部分内存，但不会让查询头本身消失。增量推理与缓存一致性验证已列在[大模型学习路线](70_机器学习到智能体学习路线.md)。

忽略批次维度时，单层缓存的元素数为 $2TH_{kv}d_h$；$T$ 是已缓存 token 数，$H_{kv}$ 是键值头数，$d_h$ 是每个头的维度，因子 $2$ 表示同时保存键和值。若数据类型每个元素占 $b$ 字节，$L$ 层模型的缓存约占 $2LTH_{kv}d_hb$ 字节。保持其他量不变，把 $H_{kv}$ 从 $4$ 降到 $2$ 会让缓存空间减半；实际总内存还包括模型参数、临时张量和运行时开销。

## 5. 建议的对照实验

1. 阅读 [`network_forward`](../src/ml/12_tiny_transformer.c#L465)，记录学习式位置向量、LayerNorm、GELU 和注意力头的基线形状。
2. 先用极小输入单独验证 RoPE 或 SwiGLU 的前向数值，再做中心差分（central difference）梯度检查（gradient check）。
3. 用 $4$ 个查询头和 $2$ 个键值头手工标出 GQA 的共享关系，并计算缓存大小比例。
4. 只有在各组件的数值检查通过后，才用相同语料、初始化种子和训练预算观察验证损失。

现有案例很小，单次训练分数容易受初始化和数据规模影响。组件对照的首要目标是确认计算、形状和资源差异，不能据此断言某个变体对所有任务都更好。
