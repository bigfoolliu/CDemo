# 阶段 7：纯 C 小型 Transformer 源码与数学推导

源码：[src/ml/12_tiny_transformer.c](../src/ml/12_tiny_transformer.c)。本案例用 C23 和 `<math.h>` 实现一个单层 decoder-only 字符模型，手写注意力、LayerNorm、GELU、交叉熵、完整反向传播和 Adam；不使用机器学习框架或自动求导。它适合逐步观察核心计算，不代表可训练实用语言模型。

![自回归 Transformer 的因果注意力矩阵](images/58_小型Transformer.svg)

第 $t$ 个位置只能读取自身和之前的位置；灰色格表示未来信息被遮挡，避免训练时偷看答案。

## 1. 运行与调试

~~~~sh
make tiny-transformer
make test-tiny-transformer
make sanitize-tiny-transformer
./build/ml/12_tiny_transformer --log
less logs/ml/12_tiny_transformer.log
~~~~

`make tiny-transformer` 运行梯度与边界断言、微型拟合、正式训练、测试集评估、生成和注意力矩阵。`make test-tiny-transformer` 走相同计算路径，但省略训练进度与矩阵；`make sanitize-tiny-transformer` 以 AddressSanitizer 和 UndefinedBehaviorSanitizer 重新运行验证。`--log` 同时输出到终端并追加写入 `logs/ml/12_tiny_transformer.log`。

在 VS Code 中选择 `C: ml/12_tiny_transformer.c` 并按 F5。若要观察正式训练，可在 `main` 调用 `train` 的位置设置断点；入口会先完成数值梯度检查，再停在训练调用处。阅读前向时重点观察 `Forward` 的 `x`、`u`、`q`、`k`、`v`、`attention`、`c`、`probability`；阅读反向时观察 `network_backward` 的 `dz`、`dy`、`du`、`dq`、`dk`、`dv`。函数与结构体位置可从[源码地图](../src/ml/12_tiny_transformer.c#L64)开始。

## 2. 先建立源码地图

| 概念 | 源码位置 | 关键状态 | 作用 |
| --- | --- | --- | --- |
| 词表和语料 | [`alphabet` 与数据集](../src/ml/12_tiny_transformer.c#L52) | `training`、`validation`、`testing` | 四字符预测任务与固定分组 |
| 样本编码 | [`dataset_encode`](../src/ml/12_tiny_transformer.c#L150) | `Sequence.input`、`Sequence.target`、`steps` | 将字符串变成“当前字符 → 下一个字符”训练对 |
| 参数与缓存 | [`Network`、`Forward`、`Adam`](../src/ml/12_tiny_transformer.c#L75) | `parameters`、各层激活、优化器矩 | 以固定数组储存模型、前向中间量和优化器状态 |
| 前向计算 | [`network_forward`](../src/ml/12_tiny_transformer.c#L429) | `x` 到 `probability` | 从字符 ID 计算每个位置的下一个字符概率 |
| 反向计算 | [`network_backward`](../src/ml/12_tiny_transformer.c#L493) | `gradient` 与局部导数数组 | 沿计算图反向累计全部参数梯度 |
| 优化与检查点 | [`train`](../src/ml/12_tiny_transformer.c#L673) | `batch`、`gradient_norm`、`best` | 全批量训练、裁剪梯度并按验证损失保留模型 |
| 生成与展示 | [`generate`](../src/ml/12_tiny_transformer.c#L767)、[`show_attention`](../src/ml/12_tiny_transformer.c#L792) | `text`、`attention` | 贪心生成字符，并打印每个头的因果注意力矩阵 |
| 入口与正确性验证 | [`main`](../src/ml/12_tiny_transformer.c#L1060)、[`test_gradients`](../src/ml/12_tiny_transformer.c#L816)、[`test_boundaries`](../src/ml/12_tiny_transformer.c#L926) | 数值差分、断言 | 先验证导数与边界，再训练、评估并生成 |

下文统一使用零起始索引：时间位置为 $t$，被关注的键位置为 $s$，特征位置为 $i$ 或 $d$，词表类别为 $k$。模型维数为 $D=16$，头数为 $H=2$，每头维数为 $d_h=D/H=8$，词表大小为 $\mathcal{V}=4$，一个样本的预测步数为 $T$。除特别说明外，向量按列理解，代码数组按行优先展开。

## 3. 模型要预测什么

词表是 `alphabet` 中的 `a`、`b`、`c` 和空格，字符 ID 分别是 $0,1,2,3$。字符串 `abac acbc ` 长度为 $10$，会产生 $9$ 个训练对：

| 位置 $t$ | 输入 `input[t]` | 目标 `target[t]` |
| ---: | --- | --- |
| $0$ | `a` | `b` |
| $1$ | `b` | `a` |
| $2$ | `a` | `c` |
| $3$ | `c` | 空格 |
| $4$ | 空格 | `a` |

所以位置 $t=2$ 的模型输入前缀是 `aba`，训练目标是 `c`。位置 $t$ 只能读取当前位置及更早字符，再预测第 $t+1$ 个字符。`dataset_encode` 对每个完整字符串独立编码，不让字符对越过字符串边界；每 $16$ 步切一块，余数保留。

本案例的单层解码器按以下顺序处理每个样本：

$$
\text{字符嵌入}+\text{位置嵌入}
\to \text{LayerNorm}
\to Q,K,V
\to \text{因果多头注意力}
\to \text{残差}
\to \text{LayerNorm}
\to \text{前馈层与 GELU}
\to \text{残差}
\to \text{LayerNorm}
\to \text{字符概率}
\tag{1}
$$

每个时间位置独立执行前馈层和分类层；只有注意力会把不同位置的信息汇入当前表示。这里没有 dropout、多个 Transformer block、旋转位置编码、KV 缓存或动态内存分配。

## 4. 参数布局与 C 数组

`Network.parameters` 是长度为 $2644$ 的连续 `double` 数组；枚举中的偏移量把各个张量切成不重叠区间。全连接权重按 [输出][输入] 存放，代码用 `w[i * inputs + j]` 访问 $W_{ij}$。

| 参数组 | 起始偏移 | 形状与数量 |
| --- | ---: | --- |
| 字符嵌入 `EMB` | $0$ | $4\times16=64$ |
| 位置嵌入 `POS` | $64$ | $16\times16=256$ |
| 第一组归一化 `G1`、`B1` | $320$ | $16+16=32$ |
| 查询投影 `QW`、`QB` | $352$ | $16\times16+16=272$ |
| 键投影 `KW`、`KB` | $624$ | $16\times16+16=272$ |
| 值投影 `VW`、`VB` | $896$ | $16\times16+16=272$ |
| 注意力输出 `OW`、`OB` | $1168$ | $16\times16+16=272$ |
| 第二组归一化 `G2`、`B2` | $1440$ | $16+16=32$ |
| 前馈第一层 `F1W`、`F1B` | $1472$ | $32\times16+32=544$ |
| 前馈第二层 `F2W`、`F2B` | $2016$ | $16\times32+16=528$ |
| 最终归一化 `G3`、`B3` | $2544$ | $16+16=32$ |
| 字符分类器 `UW`、`UB` | $2576$ | $4\times16+4=68$ |

合计为 $64+256+32+4\times272+544+528+32+68=2644$ 个可训练参数。Q/K/V 投影各先生成 $16$ 维向量，再按维度切成两个 $8$ 维头；不是分别维护六个独立的头投影矩阵。[`network_initialize`](../src/ml/12_tiny_transformer.c#L192) 用固定种子初始化嵌入和权重到 $[-0.3,0.3)$，LayerNorm 缩放参数初始为 $1$、平移参数初始为 $0$，因此运行可复现。

## 5. 沿一个位置完整走前向计算

### 5.1 字符与位置嵌入

设输入字符 ID 为 $\xi_t$，其中 $0\le\xi_t<\mathcal{V}$，字符嵌入矩阵为 $E\in\mathbb{R}^{\mathcal{V}\times D}$，可学习位置矩阵为 $P\in\mathbb{R}^{16\times D}$，则位置表示为：

$$
\begin{aligned}
x_{t,i}&=E_{\xi_t,i}+P_{t,i},\\
0&\le t<16,\\
0&\le i<16
\end{aligned}
\tag{2}
$$

代码对应 `forward->x[...] = p[EMB + ...] + p[POS + ...]`。例如真实训练后的位置 $t=2$ 输入 `a`，第一个维度的两张表相加得到 $x_{2,0}\approx-0.006832665486$。字符嵌入告诉网络“这是 `a`”，位置嵌入告诉网络“这是上下文中的第 $3$ 个字符”；二者相加后才送入注意力层。

### 5.2 全连接投影

[`dense_forward`](../src/ml/12_tiny_transformer.c#L222) 对每个位置独立执行仿射变换。用大写 $X,Y$ 表示任一全连接层的输入与输出，若输入宽度为 $I$、输出宽度为 $O$，则：

$$
\begin{aligned}
Y_{t,i}&=b_i+\sum_{j=0}^{I-1}W_{i,j}X_{t,j},\\
0&\le i<O
\end{aligned}
\tag{3}
$$

此处大写 $X_{t,j}$ 是任一全连接层的输入，$W_{i,j}$ 是第 $i$ 个输出对第 $j$ 个输入的权重，$b_i$ 是偏置，结果写入该层输出。矩阵形状是 $O\times I$，时间维度不被混合。

手算一个 $2\to1$ 的例子：令标量层输入向量为 $X=(2,-1)$、权重为 $(0.5,-2)$、偏置为 $0.25$，则 $y=0.25+0.5(2)+(-2)(-1)=3.25$。若这个输出收到上游梯度 $\delta=0.4$，参数梯度为 $(0.8,-0.4)$，偏置梯度为 $0.4$，输入梯度为 $(0.2,-0.8)$。这正是 Q/K/V、注意力输出、前馈层和分类器共享的 `dense_forward` / `dense_backward` 计算。

### 5.3 LayerNorm：每个位置沿特征归一化

对一个位置的 $D$ 个特征，LayerNorm 计算总体均值 $\mu_t$、总体方差 $\sigma_t^2$、逆标准差 $r_t$ 和标准化特征 $\hat{x}_{t,i}$：

$$
\begin{aligned}
\mu_t&=\frac{1}{D}\sum_{i=0}^{D-1}x_{t,i},\\
\sigma_t^2&=\frac{1}{D}\sum_{i=0}^{D-1}(x_{t,i}-\mu_t)^2
\end{aligned}
\tag{4}
$$

$$
\begin{aligned}
r_t&=\frac{1}{\sqrt{\sigma_t^2+\varepsilon}},\\
\hat{x}_{t,i}&=(x_{t,i}-\mu_t)r_t,\\
u_{t,i}&=\gamma_i\hat{x}_{t,i}+\beta_i
\end{aligned}
\tag{5}
$$

$\varepsilon=10^{-5}$ 防止常量输入时除以零；$\gamma_i$ 和 $\beta_i$ 是可学习的缩放与平移。[`norm_forward`](../src/ml/12_tiny_transformer.c#L274) 对每个 $t$ 单独遍历 `DIM` 个特征，不会把不同时间位置放在一起求平均。三组 LayerNorm 分别对应 `G1/B1`、`G2/B2`、`G3/B3`。

在上面的真实样本位置 $t=2$，完整 $16$ 维输入得到 $\mu_2\approx-0.037026432914$、$\sigma_2^2\approx0.034998425685$、$r_2\approx5.344581566084$。对特征 $i=0$：

$$
\hat{x}_{2,0}
=\frac{-0.006832665486-(-0.037026432914)}
{\sqrt{0.034998425685+0.00001}}
\approx0.161373052809
\tag{6}
$$

训练后该维 $\gamma_0\approx0.966768225497$、$\beta_0\approx0.010793912478$，因此 $u_{2,0}\approx0.966768225497(0.161373052809)+0.010793912478=0.166804252386$。这也说明 LayerNorm 的输出不是简单的零均值单位方差：学习到的 $\gamma$ 和 $\beta$ 会再调整它。

### 5.4 Q、K、V 与缩放点积注意力

第一组 LayerNorm 输出 $u$ 分别通过三个独立投影得到查询 $Q$、键 $K$、值 $V$：

$$
\begin{aligned}
Q&=uW_Q^\mathsf{T}+b_Q,\\
K&=uW_K^\mathsf{T}+b_K,\\
V&=uW_V^\mathsf{T}+b_V
\end{aligned}
\tag{7}
$$

其中 $W_Q,W_K,W_V\in\mathbb{R}^{D\times D}$，$b_Q,b_K,b_V\in\mathbb{R}^{D}$；每个头只读取自己负责的 $8$ 个特征。查询表示当前位置要找什么，键表示每个历史位置可供匹配的内容，值表示匹配后实际取回的信息。对于头 $h$、查询位置 $t$ 和键位置 $s$，缩放分数为：

$$
S^{(h)}_{t,s}
=\frac{\sum_{i=0}^{d_h-1}
Q_{t,\,h d_h+i}K_{s,\,h d_h+i}}{\sqrt{d_h}}
\tag{8}
$$

除以 $\sqrt{d_h}$ 可控制点积随特征数增长的幅度。这里头索引满足 $0\le h<H$，键索引满足 $0\le s\le t<T$。因 $d_h=8$，缩放因子是 $1/\sqrt{8}\approx0.353553$。对每个头 $h$ 和查询位置 $t$，稳定计算使用该行最大有效分数 $S^{(h)}_{\max,t}=\max_{0\le s'\le t}S^{(h)}_{t,s'}$。下表给出真实训练后样本 `abac acbc ` 在查询位置 $t=2$、第 $0$ 个头的分数；原始点积未缩放，稳定指数为 $\exp(S^{(0)}_{2,s}-S^{(0)}_{\max,2})$：

| 键位置 $s$ | $Q_2^{(0)}\cdot K_s^{(0)}$ | $S^{(0)}_{2,s}$ | 稳定指数 $\exp(S-S_{\max})$ | $A^{(0)}_{2,s}$ |
| ---: | ---: | ---: | ---: | ---: |
| $0$ | $1.153151$ | $0.407701$ | $0.859662$ | $0.410272$ |
| $1$ | $1.580853$ | $0.558916$ | $1.000000$ | $0.477248$ |
| $2$ | $-2.506942$ | $-0.886338$ | $0.235686$ | $0.112481$ |

对该查询行，$S^{(0)}_{\max,2}=\max_{s\le2}S^{(0)}_{2,s}=0.558916$。减去最大值不会改变 softmax 结果，但能避免指数过大。指数分母为 $0.859662+1+0.235686=2.095349$；以键位置 $s=0$ 为例，权重是 $0.859662/2.095349\approx0.410272$。三项权重和为 $1$。

因果掩码是在归一化前生效的。[`attention_forward`](../src/ml/12_tiny_transformer.c#L339) 只遍历 `s <= t`，因此查询 $t=2$ 只能看到键位置 $0,1,2$；所有 $s>2$ 的权重保持为严格的 $0$。若先对未来位置做 softmax 再把概率清零，未来分数已经进入分母，会改变过去位置的权重并造成信息泄漏。

一个头的上下文是允许位置的值向量加权和；对头内特征索引 $0\le i<d_h$，两个头分别写入输出向量的不同特征切片，相当于拼接：

$$
\begin{aligned}
C_{t,\,h d_h+i}&=\sum_{s=0}^{t}A^{(h)}_{t,s}V_{s,\,h d_h+i},\\
A^{(h)}_{t,s}
&=\frac{\exp(S^{(h)}_{t,s}-S^{(h)}_{\max,t})}
{\sum_{j=0}^{t}\exp(S^{(h)}_{t,j}-S^{(h)}_{\max,t})}
\end{aligned}
\tag{9}
$$

通用公式中的行最大值定义为 $S^{(h)}_{\max,t}=\max_{0\le s'\le t}S^{(h)}_{t,s'}$。在同一真实样本中，第 $0$ 个头、第 $0$ 个输出特征的三个加权项依次为 $0.410272(0.403602)=0.165587$、$0.477248(0.210530)=0.100475$、$0.112481(-0.670343)=-0.075401$，所以 $C_{2,0}\approx0.190661$。接着 `OW/OB` 再把拼接的 $16$ 维上下文投影回 $16$ 维。

从训练完成并恢复最佳验证检查点后的同一条样本，可以把第 $t=2$ 个位置贯穿整个网络。下表列出特征 $i=0$ 的真实中间值；`pre` 与 GELU 激活宽度为 $32$，其他列对应 $16$ 维状态：

| 阶段 | 对应状态 | 特征 $0$ 的值 |
| --- | --- | ---: |
| 字符与位置嵌入 | $x_{2,0}$ | $-0.006833$ |
| 第一组 LayerNorm | $u_{2,0}$ | $0.166804$ |
| 第 $0$ 头注意力上下文 | $C_{2,0}$ | $0.190661$ |
| 注意力输出投影 | $o_{2,0}$ | $-0.223265$ |
| 第一条残差 | $y_{2,0}=x_{2,0}+o_{2,0}$ | $-0.230098$ |
| 第二组 LayerNorm | $m_{2,0}$ | $-0.128449$ |
| 前馈第一层 | $a_{2,0}$ | $-0.329735$ |
| GELU | $\tilde a_{2,0}$ | $-0.122266$ |
| 前馈第二层 | $f_{2,0}$ | $-0.650318$ |
| 第二条残差 | $z_{2,0}=y_{2,0}+f_{2,0}$ | $-0.880415$ |
| 最终 LayerNorm | $\text{last}_{2,0}$ | $-1.543670$ |

例如两条残差分别是 $-0.006832665486-0.223265171591=-0.230097837077$ 和 $-0.230097837077-0.650317639028=-0.880415476105$。第二组 LayerNorm 的均值为 $-0.181771044333$、方差为 $0.141340372322$、逆标准差为 $2.659815555629$；特征 $0$ 的标准化值为 $(-0.230097837077+0.181771044333)2.659815555629=-0.128540355094$，再经 $\gamma_0=1.029626965155$、$\beta_0=0.003899546921$ 得到 $m_{2,0}=-0.128449068796$。最终 LayerNorm 同理：输入 $z$ 的均值为 $-0.013046502040$、方差为 $0.448003868581$、逆标准差为 $1.494012637030$；标准化特征为 $(-0.880415476105+0.013046502040)1.494012637030=-1.295860208221$，再乘 $\gamma_0=1.128388372459$ 并加 $\beta_0=-0.081436660895$，得到 `last[2 * DIM] = -1.543670252185`。

入口也包含一个完全手算的单头边界例子：令所有 $Q=K=0$，各位置所有 $V$ 特征依次为 $1,3,5$，每个允许分数都相同。于是：

$$
\begin{aligned}
A&=
\begin{bmatrix}
1&0&0\\
1/2&1/2&0\\
1/3&1/3&1/3
\end{bmatrix},\\
C&=
\begin{bmatrix}
1\\2\\3
\end{bmatrix}
\end{aligned}
\tag{10}
$$

这验证了单头注意力的因果范围和加权平均。正式模型的两头矩阵由 `show_attention` 打印；行是查询位置，列是键位置。注意力权重显示路由强度，例如同一查询 $t=2$ 时，第 $0$ 头权重约为 $(0.410,0.477,0.112)$，第 $1$ 头约为 $(0.501,0.323,0.177)$；两个头使用不同的投影与特征切片。单独一张权重图不能证明模型形成了人类意义上的解释。

### 5.5 残差、GELU 与输出分类

注意力输出投影加回原表示，构成第一条残差；随后第二次 LayerNorm、两层前馈和第二条残差：

$$
\begin{aligned}
y&=x+o,\\
m&=\operatorname{LN}_2(y),\\
a&=mW_1^\mathsf{T}+b_1
\end{aligned}
\tag{11}
$$

$$
\begin{aligned}
\tilde a&=\operatorname{GELU}(a),\\
f&=\tilde aW_2^\mathsf{T}+b_2,\\
z&=y+f
\end{aligned}
\tag{12}
$$

$x,o,y,m,f,z\in\mathbb{R}^{T\times16}$，而 $a,\tilde a\in\mathbb{R}^{T\times32}$。例如在一个仅用于说明残差的标量例子中，输入 $x=2$、分支输出 $o=-0.5$，相加得到 $y=1.5$；反向时 $y$ 的导数会同时沿恒等路径和 $o$ 分支传播。

源码采用精确的 `erf` 形式，而不是常见的 `tanh` 近似。令标准正态分布的累积分布函数和密度函数分别为 $\Phi$ 与 $\phi$，则：

$$
\begin{aligned}
\operatorname{GELU}(x)=x\Phi(x)
&=\frac{x}{2}\left(1+\operatorname{erf}\left(\frac{x}{\sqrt{2}}\right)\right)
\end{aligned}
\tag{13}
$$

$$
\begin{aligned}
\frac{d\,\operatorname{GELU}(x)}{dx}
&=\Phi(x)+x\phi(x),\\
\phi(x)&=\frac{\exp(-x^2/2)}{\sqrt{2\pi}}
\end{aligned}
\tag{14}
$$

例如 $x=1$ 时，$\Phi(1)\approx0.841345$、$\phi(1)\approx0.241971$，所以 $\operatorname{GELU}(1)\approx0.841345$，导数约为 $1.083315$。前向的 `activation` 保存 GELU 输出；反向从 `pre` 重新计算同一导数，再逐元素乘上游梯度。

第三组 LayerNorm 的输出存入 `last`。分类器权重 $W^U\in\mathbb{R}^{\mathcal{V}\times D}$、偏置 $b^U\in\mathbb{R}^{\mathcal{V}}$ 将每个位置映射到 $\mathcal{V}=4$ 个 logits。为避免与残差输出 $z$ 混淆，记第 $t$ 个位置、第 $k$ 类的 logit 为 $\zeta_{t,k}$：

$$
\begin{aligned}
\zeta_{t,k}&=b^U_k+\sum_{i=0}^{D-1}W^U_{k,i}\operatorname{last}_{t,i},\\
0&\le k<\mathcal{V}
\end{aligned}
\tag{15}
$$

真实样本位置 $t=2$、目标类别 `c` 的代入结果为 $\zeta_{2,c}=3.673161$；四类 logits 的完整结果见下方 softmax 算例。对类别 $k$，稳定 softmax 先令 $M_t=\max_j \zeta_{t,j}$，再计算：

$$
p_{t,k}=\frac{\exp(\zeta_{t,k}-M_t)}{\sum_{j=0}^{\mathcal{V}-1}\exp(\zeta_{t,j}-M_t)}
\tag{16}
$$

这里 $p_{t,k}$ 是模型在位置 $t$ 预测下一个字符为类别 $k$ 的概率。前向把 `probability` 数组先用作 logits 的输出缓冲，计算后再原地覆盖为概率。令 $y_t^\star$ 表示 `target[t]` 的字符类别；上标星号用来和残差张量 $y$ 区分。损失用交叉熵和稳定的 log-sum-exp 计算：

$$
\begin{aligned}
\ell_t&=\log\left(\sum_{k=0}^{\mathcal{V}-1}\exp(\zeta_{t,k}-M_t)\right)+M_t-\zeta_{t,y_t^\star}
=-\log p_{t,y_t^\star},\\
L&=\frac{1}{T}\sum_{t=0}^{T-1}\ell_t
\end{aligned}
\tag{17}
$$

真实样本位置 $t=2$ 的目标是 `c`，恢复第 $50$ 轮最佳验证检查点后，概率为 $p_a=0.004391$、$p_b=0.028159$、$p_c=0.955606$、$p_{\text{空格}}=0.011844$。该位置的损失为 $-\log(0.955606)\approx0.045410$；整段 $9$ 步的平均损失为 $0.231163$。

该位置的实际 logits（类别顺序 `a,b,c,空格`）为 $(-1.709670,0.148683,3.673161,-0.717347)$，最大值 $M_2=3.673161$。减去最大值后取指数，得到 $(0.004595,0.029467,1,0.012394)$，总和为 $1.046456$；因此 $p_{2,c}=1/1.046456=0.955606$。这把稳定 softmax 的减最大值、求指数、归一化三步与源码中的数值对应起来。

## 6. 反向传播如何对应代码

反向传播把损失对输出的影响沿前向计算顺序反向传递。记 $g^X=\partial L/\partial X$ 为损失对张量 $X$ 的导数；代码里的 `dx`、`dy`、`dq` 等数组保存这些导数。共享参数在所有时间位置上重复使用，所以每个位置产生的权重和偏置导数要累加。

### 6.1 softmax 与交叉熵的合并导数

在类别 $k$ 和位置 $t$ 上，softmax 和交叉熵合并后，logit 导数可直接写为：

$$
\frac{\partial L}{\partial \zeta_{t,k}}
=\frac{p_{t,k}-\mathbf{1}[k=y_t^\star]}{T}
\tag{18}
$$

$\mathbf{1}[\cdot]$ 在条件成立时为 $1$，否则为 $0$；除以 $T$ 是因为 [`network_forward`](../src/ml/12_tiny_transformer.c#L429) 返回位置平均损失。位置 $t=2$ 的真实概率代入后，类别顺序为 `a,b,c,空格`：

$$
g^{\zeta}_{2}
\approx(0.000488,\ 0.003129,\ -0.004933,\ 0.001316)
\tag{19}
$$

目标类别 `c` 的梯度为负，梯度下降会提高其 logit；其他类别的梯度为正，更新会降低它们的 logit。随后 [`dense_backward`](../src/ml/12_tiny_transformer.c#L248) 将该导数传给分类器权重、偏置和最终 LayerNorm。

### 6.2 全连接层的导数

对 $Y=XW^\mathsf{T}+b$，输入有 $T$ 行、宽度为 $I$，输出宽度为 $O$，其梯度为：

$$
\begin{aligned}
g^W_{i,j}&=\sum_{t=0}^{T-1}g^Y_{t,i}X_{t,j},\\
g^b_i&=\sum_{t=0}^{T-1}g^Y_{t,i},\\
g^X_{t,j}&=\sum_{i=0}^{O-1}W_{i,j}g^Y_{t,i}
\end{aligned}
\tag{20}
$$

在前面的手算例中，$X=(2,-1)$、单个输出梯度为 $0.4$，因此 $g^W=(0.8,-0.4)$、$g^b=0.4$、$g^X=(0.2,-0.8)$。代码的外层时间循环实现求和；`dw += delta * x` 和 `dx += weight * delta` 正对应上式。

### 6.3 注意力的导数

对一个头、一个查询位置，先从上下文加权和 $C_{t,d}=\sum_s A_{t,s}V_{s,d}$ 求导。设 $g^C_{t,d}$ 是该上下文特征收到的上游导数，则：

$$
\begin{aligned}
g^V_{s,d}\mathrel{+}&=A_{t,s}g^C_{t,d},\\
g^A_{t,s}&=\sum_d g^C_{t,d}V_{s,d}
\end{aligned}
\tag{21}
$$

[`attention_backward`](../src/ml/12_tiny_transformer.c#L389) 计算 softmax 行导数；令 $g^A_{t,s}$ 为损失对注意力权重的导数，可整理为：

$$
\begin{aligned}
g^S_{t,s}
&=A_{t,s}\left(g^A_{t,s}
-\sum_{j=0}^{t}A_{t,j}g^A_{t,j}\right)
\end{aligned}
\tag{22}
$$

最后把分数对查询和键的导数传回：

$$
\begin{aligned}
g^Q_{t,d}\mathrel{+}&=\frac{1}{\sqrt{d_h}}\sum_{s=0}^{t}g^S_{t,s}K_{s,d},\\
g^K_{s,d}\mathrel{+}&=\frac{1}{\sqrt{d_h}}g^S_{t,s}Q_{t,d}
\end{aligned}
\tag{23}
$$

手算 softmax 部分：若一行权重为 $(0.25,0.75)$，上游的 $g^A$ 为 $(2,0)$，加权均值为 $0.25(2)+0.75(0)=0.5$，所以未缩放的 $g^S$ 为 $(0.25(2-0.5),0.75(0-0.5))=(0.375,-0.375)$。若使用正式模型每头的缩放 $1/\sqrt{8}$，传向点积的系数变为约 $(0.132583,-0.132583)$，再分别乘对应的键或查询特征。源码把 $1/\sqrt{d_h}$ 合并到局部变量 `ds`，因此与上式把缩放写在 Q/K 导数外完全等价。[`attention_backward`](../src/ml/12_tiny_transformer.c#L389) 只遍历 $s\le t$，因果掩码边不会获得梯度；同一个键会被多个后续查询使用，因此 `dk` 和 `dv` 要累加。

### 6.4 LayerNorm 的导数

对单个位置，[`norm_backward`](../src/ml/12_tiny_transformer.c#L307) 令 $g^u_i$ 表示归一化输出收到的上游导数，$\eta_i=g^u_i\gamma_i$，$D=16$。把均值和方差对输入的影响合并后：

$$
\begin{aligned}
g^x_i&=r_t\left(\eta_i-\frac{1}{D}\sum_j \eta_j
-\frac{\hat{x}_i}{D}\sum_j \eta_j\hat{x}_j\right)
\end{aligned}
\tag{24}
$$

$$
\begin{aligned}
g^{\gamma}_i\mathrel{+}&=g^u_i\hat{x}_i,\\
g^{\beta}_i\mathrel{+}&=g^u_i
\end{aligned}
\tag{25}
$$

这里 $r_t$ 是前向缓存的逆标准差，$\eta_i$ 是缩放后的上游导数。源码局部变量 `inverse[t]`、`value`、`sum`、`projection` 分别保存 $r_t$、$\eta_i$、$\sum_i\eta_i$ 和 $\sum_i\eta_i\hat{x}_i$。

边界测试给 `x` 的 $16$ 个特征全赋值 $5$，令 $\gamma_i=1$、$\beta_i=0$、上游梯度 $g^u_i=i$。此时均值为 $5$、方差为 $0$、$\hat{x}_i=0$、逆标准差为 $1/\sqrt{10^{-5}}\approx316.227766$，所以：

$$
\begin{aligned}
g^x_i&=316.227766\left(i-\frac{0+1+\cdots+15}{16}\right)
=316.227766(i-7.5)
\end{aligned}
\tag{26}
$$

例如 $g^x_0\approx-2371.708245$，$g^x_{15}\approx2371.708245$。梯度幅度较大是 $\varepsilon$ 很小且输入方差为零的结果；测试确认它仍然是有限值，而不是 NaN 或无穷大。

### 6.5 GELU、残差与嵌入梯度

GELU 的反向使用前述导数：对每个前馈输入 $a_i$，将上游导数乘以 $\Phi(a_i)+a_i\phi(a_i)$。例如 $a_i=1$ 时，比例约为 $1.083315$；$a_i=0$ 时，比例恰为 $0.5$。这对应 `network_backward` 中逐个重算 `pre` 的导数并修改 `da`。

残差 $y=x+o$ 对两条输入的导数都是 $g^y$；第二条残差 $z=y+f$ 则使 $g^y=g^z+g^f$。例如上游直接导数 $g^z=0.3$、前馈分支导数 $g^f=-0.1$ 时，回到 $y$ 的导数为 $0.3+(-0.1)=0.2$。在代码中，前馈支路经 `norm_backward` 得到的 `dy` 再加上直通导数 `dz`；随后注意力支路反传得到另一个 `branch`，最终输入导数为 `dx = dy + branch`。Q、K、V 都由同一个归一化张量 $u$ 投影而来，因此它们回传到 $u$ 的三份导数也必须相加，代码先把 K、V 分支逐项加进 `du`。

字符嵌入在重复字符处被共享。例如一个样本里 `a` 出现多次，属于同一个 `EMB` 向量的梯度必须把这些位置的贡献相加。位置嵌入则按位置独立，因此每个时刻的总输入导数累加到该时刻对应的 `POS` 行。

## 7. 从初始化到一次完整训练

### 7.1 数据分组与块

正式数据训练、验证、测试字符串已分别固定，数量为 $8,2,2$；各组分别切上下文块，避免同一原始字符串的不同块分落到不同数据组。每条字符串长 $30$，会有 $29$ 个“当前字符—下一个字符”预测对；所以一条训练字符串切为 $16$ 步和 $13$ 步两块。数据集共得到 $16,4,4$ 块，训练样本共有 $8\times29=232$ 个预测位置，验证和测试各有 $2\times29=58$ 个位置。块之间不传递隐藏状态，字符对不重叠，尾块不会丢弃。

### 7.2 全批量梯度、裁剪与 Adam

[`network_backward`](../src/ml/12_tiny_transformer.c#L493) 为每块输出该块内部的平均梯度。若第 $s$ 块有 $T_s$ 个位置、平均梯度为 $g_s$，则合并成按字符数平均的全批量梯度：

$$
g_{\text{batch}}
=\frac{\sum_s T_s g_s}{\sum_s T_s}
\tag{27}
$$

例如同一条 $29$ 位置字符串分成 $16$ 步与 $13$ 步两块，其合并结果是 $(16g_{16}+13g_{13})/29$，不能直接对两个块的平均梯度再除以 $2$。完整训练集再把所有 $16$ 块按各自长度加权，合计 $232$ 个位置。

[`clip_gradient`](../src/ml/12_tiny_transformer.c#L566) 先计算所有 $2644$ 个参数梯度的 L2 范数 $G=\sqrt{\sum_i g_i^2}$，其中 $g_i$ 是第 $i$ 个参数的梯度；若超过阈值 $\tau=1$，所有分量乘 $\tau/G$，方向不变。测试用梯度 $(3,4)$ 验证：范数为 $5$，裁剪后为 $(0.6,0.8)$。代码先扫描并验证全部梯度有限，失败时不修改数组。

Adam 用一阶矩 $m_n$、二阶矩 $v_n$ 和偏差修正更新单个参数 $\theta_n$；$n$ 是优化器更新次数，$g_n$ 是该参数在第 $n$ 次更新时收到的梯度，不是字符位置。这里 Adam 的一阶矩 $m_n$ 与 `Forward.m`（第二次 LayerNorm 输出）是不同数组；LayerNorm 平移参数 $\beta_i$ 也与 Adam 衰减系数 $\beta_1,\beta_2$ 不同。学习率为 $0.003$，$\beta_1=0.9$、$\beta_2=0.999$、$\epsilon_{\text{Adam}}=10^{-8}$：

$$
\begin{aligned}
m_n&=\beta_1m_{n-1}+(1-\beta_1)g_n,\\
v_n&=\beta_2v_{n-1}+(1-\beta_2)g_n^2
\end{aligned}
\tag{28}
$$

$$
\begin{aligned}
\hat{m}_n&=\frac{m_n}{1-\beta_1^n},\\
\hat{v}_n&=\frac{v_n}{1-\beta_2^n},\\
\theta_n&=\theta_{n-1}
-0.003\frac{\hat{m}_n}{\sqrt{\hat{v}_n}+10^{-8}}
\end{aligned}
\tag{29}
$$

以首次更新 $n=1$ 的梯度 $g_1=0.2$、初始矩为零为例，$m_1=0.02$、$v_1=0.00004$，偏差修正后 $\hat m_1=0.2$、$\hat v_1=0.04$，参数减少约 $0.003$。[`adam_update`](../src/ml/12_tiny_transformer.c#L598) 会先在临时副本中计算并检查所有候选参数和矩；任一非有限值都会让整次更新失败，不会部分提交。

每轮先对训练集做一次完整批量更新，再用验证集评估；训练梯度不读取验证或测试样本。`best` 只在验证交叉熵下降时更新，训练结束后恢复验证分数最佳的参数。测试集只在训练结束后用于一次独立评估。

### 7.3 运行结果该如何理解

以下数值来自固定种子 $42$；训练后恢复验证损失最佳的第 $50$ 轮检查点，再运行测试评估。训练进度日志的关键点为：

| 轮次 | 训练 CE | 验证 CE | 裁剪前梯度范数 |
| ---: | ---: | ---: | ---: |
| 初始 | $1.54140$ | — | — |
| $1$ | $1.41164$ | $1.36699$ | $1.58028$ |
| $50$ | $0.28558$ | $0.39055$ | $0.22295$ |
| $100$ | $0.14163$ | $0.68163$ | $0.52170$ |
| $150$ | $0.12375$ | $0.87428$ | $0.73403$ |
| $200$ | $0.12246$ | $1.01284$ | $1.05759$ |
| $250$ | $0.11971$ | $1.06561$ | $0.23223$ |

`gradient_norm` 是裁剪前的范数；因此某轮大于阈值 $1$ 表示梯度向量会被整体缩放。

[`evaluate`](../src/ml/12_tiny_transformer.c#L643) 与 [`bigram_evaluate`](../src/ml/12_tiny_transformer.c#L725) 产生以下独立评估：

| 评估项目 | 交叉熵 | 下一字符准确率 |
| --- | ---: | ---: |
| 周期微型语料拟合 | $0.00365$ | $100\%$ |
| Transformer 独立测试集 | $0.41237$ | $79.31\%$ |
| 加一平滑 bigram 独立测试集 | $0.75419$ | $60.34\%$ |

Transformer 保留第 $50$ 轮的验证检查点，验证交叉熵为 $0.39055$；第 $250$ 轮训练交叉熵降至 $0.11971$，验证交叉熵却升到 $1.06561$，说明继续贴合训练集会过拟合。结果只针对固定的小型合成语料，用于验证实现与比较基线，不能证明模型能理解一般语言或泛化到真实文本。

bigram 基线只根据当前字符统计下一个字符。若某当前字符的四类计数（先加一平滑）是 $(4,2,1,1)$，则总数为 $8$，预测概率为 $(0.5,0.25,0.125,0.125)$。加一使未见过的转移仍有非零概率；这只是解释 `bigram_evaluate` 的示例计数，不是正式训练集某一行的实际统计。

## 8. 生成、边界与验证

[`generate`](../src/ml/12_tiny_transformer.c#L767) 从字符 `a` 开始，每一步选择概率最高的类别（平局时选最小 ID），把它追加到输入，再预测下一字符。输入超过上下文长度时只保留最近 $16$ 个字符，并将位置编号重新从 $0$ 开始；每一步重新计算整个上下文，没有 KV 缓存。生成时不再使用正确答案，因此一次错误会影响后续输入。固定实验生成的结果为 `[abac acbc babc acbcbcbcbacbcbcbacbac acbcbacbcbcbcbacbcbc ac]`。贪心输出展示的是模型行为，不能替代测试集指标。

测试覆盖了以下关键性质：

[`test_gradients`](../src/ml/12_tiny_transformer.c#L816) 用中心差分近似参数导数，再和反向传播结果逐项比较。对参数 $\theta_i$，步长为 $\varepsilon=10^{-5}$：

$$
\frac{\partial L}{\partial \theta_i}
\approx\frac{L(\theta_i+\varepsilon)-L(\theta_i-\varepsilon)}{2\varepsilon}
\tag{30}
$$

手算校验这种近似：若 $L(\theta)=\theta^2$、$\theta=3$、$\varepsilon=0.01$，则 $(3.01^2-2.99^2)/0.02=6$，与解析导数 $2\theta=6$ 一致。源码对 Transformer 的每个参数都做相同检查，测试样本是 `input={a,b,a}`、`target={b,a,c}`，单参数断言容差为 $2\times10^{-6}+2\times10^{-4}|g_{\text{num}}|$。

- [`test_primitive_gradients`](../src/ml/12_tiny_transformer.c#L843) 单独检查 Q/K/V 输入导数与 LayerNorm 的输入、缩放、平移导数；局部差分通过后仍另外检查完整网络梯度。
- [`test_boundaries`](../src/ml/12_tiny_transformer.c#L926) 验证单头手算、每行注意力权重和为 $1$、未来列严格为 $0$、修改未来字符不会改变过去预测、截断前缀保持相同输出、常量输入 LayerNorm 有限。
- 数据测试覆盖单字符文本、非法字符、空输入、超容量文本和尾块保留；优化器测试覆盖梯度裁剪和非有限更新时的原子失败。
- 主流程还验证周期微型拟合、训练损失下降，并要求独立测试集优于 bigram 基线。

C23 相关语法包括文件中的 `static_assert` 和空指针常量 `nullptr`；不要把它们误认为机器学习算法的一部分。数学计算使用标准数学库函数 `sqrt`、`exp`、`log`、`erf`、`fmax`；可选日志由项目记录器处理。

## 9. 建议的阅读与实验顺序

1. 先读 `dataset_encode` 和 `Sequence`，手工写出 `abac` 的输入/目标对应。
2. 在 `network_forward` 观察 `x → u → q/k/v → attention → c → y → m → activation → z → probability`，重点确认数组形状和时间索引。
3. 用打印的 `head=0, query=2` 行核对本教程中的分数、softmax 和加权值计算；再查看另一个头为何可以产生不同路由。
4. 从 `network_backward` 先追踪交叉熵到分类器，再追 GELU、LayerNorm、两条残差与 Q/K/V 三支，检查每个共享分支是否累加。
5. 运行 `make test-tiny-transformer` 查看梯度与边界验证；运行 `make tiny-transformer` 对照训练、验证、测试曲线。
6. 可尝试去掉位置嵌入、改上下文长度或头数，并重新检查维度整除、位置数组和注意力缓存；模型选择只看验证集，测试集不要用于挑参数。

后续可以在保持测试和训练数据划分的前提下研究多层堆叠、dropout、位置编码变体、KV 缓存和更大语料。每次只改变一个机制，再用数值梯度检查与因果测试确认其实现。
