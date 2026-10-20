# 阶段 1：从单个神经元到 XOR 小网络

本教程逐步拆解 [`src/ml/06_xor_mlp.c`](../src/ml/06_xor_mlp.c)：从一个手算的 sigmoid 神经元，走到含隐藏层的 XOR 网络，再核对真实初始化、前向传播、反向传播、一次参数更新和训练结果。建议先读[阶段 0：数值基础](51_numeric_foundations.md)。

## 运行与源码地图

```sh
make xor-mlp
make test-xor-mlp
make sanitize-xor-mlp
```

程序不需要输入、数据文件或外部框架。`make test-xor-mlp` 运行演示和断言；`make sanitize-xor-mlp` 使用地址与未定义行为消毒器重新构建并运行。

| 概念 | 源码 | 负责的状态或动作 |
| --- | --- | --- |
| 网络和前向缓存 | [`Network`、`Forward`](../src/ml/06_xor_mlp.c#L27-L35) | 固定数组保存参数、隐藏层输出、logit 和概率 |
| 确定性初始化 | [`network_initialize`](../src/ml/06_xor_mlp.c#L65-L91) | 权重按种子生成，偏置清零 |
| 前向传播 | [`network_forward`](../src/ml/06_xor_mlp.c#L93-L113) | 从输入依次计算隐藏层、logit 和概率 |
| 单样本反向传播 | [`network_backward`](../src/ml/06_xor_mlp.c#L115-L138) | 写入 17 个梯度，不改参数 |
| SGD 更新与平均损失 | [`network_update`、`network_loss`](../src/ml/06_xor_mlp.c#L140-L168) | 更新参数；在四个样本上重新评估损失 |
| 行为验证 | [`test_neuron` 至 `test_initialization_and_update`](../src/ml/06_xor_mlp.c#L170-L258) | 检查数值稳定性、梯度、零初始化和更新 |
| 训练入口 | [`main`](../src/ml/06_xor_mlp.c#L260-L321) | 固定顺序训练、定期输出、检查预测 |

执行顺序是：先验证单神经元和随机初始化网络的梯度，再验证零初始化与单步更新，最后训练并检查完整 XOR 真值表。训练入口的细节见 [`main` 的训练循环](../src/ml/06_xor_mlp.c#L275-L320)。

## 1. 任务、变量和网络形状

XOR 的两个输入不同则标签为 $1$，相同则标签为 $0$。样本索引为 $n\in\{0,1,2,3\}$，输入下标为 $i\in\{0,1\}$；这两个下标在源码循环中都从 $0$ 开始。

| 数学符号 | 含义 | 源码对应 |
| --- | --- | --- |
| $x_{n,i}\in\{0,1\}$ | 第 $n$ 个样本的第 $i$ 个输入 | `xor_inputs[n][i]` |
| $y_n\in\{0,1\}$ | 第 $n$ 个样本的正确标签 | `xor_targets[n]` |
| $h\in\{0,1,2,3\}$ | 隐藏神经元下标 | `for (size_t h = 0; h < HIDDEN; ++h)` |
| $W_{1,h,i}, b_{1,h}$ | 输入层到隐藏层的权重和偏置 | `parameters[W1 + h * INPUTS + i]`、`parameters[B1 + h]` |
| $u_h, a_h$ | 隐藏层第 $h$ 个神经元的加权输入与激活 | `preactivation`、`forward.hidden[h]` |
| $W_{2,h}, b_2$ | 隐藏层到输出层的权重和偏置 | `parameters[W2 + h]`、`parameters[B2]` |
| $z, p$ | sigmoid 前的 logit 和输出概率 | `forward.logit`、`forward.probability` |
| $L, J$ | 单个样本的交叉熵和四样本平均损失 | `binary_cross_entropy`、`network_loss` |

所有输入、权重、偏置、激活和损失在这个小例子中都没有物理单位。令 $\mathbb R$ 表示实数集合，网络形状为：

$$
x\in\mathbb R^2
\longrightarrow
W_1\in\mathbb R^{4\times2},\ b_1\in\mathbb R^4
\longrightarrow
a\in\mathbb R^4
\longrightarrow
W_2\in\mathbb R^4,\ b_2\in\mathbb R
\longrightarrow
z\in\mathbb R
\longrightarrow
p\in(0,1).
$$

源码把参数存入单个 `double parameters[17]`，便于统一遍历和逐项检查梯度。枚举偏移定义在 [`W1` 到 `PARAMETERS`](../src/ml/06_xor_mlp.c#L18-L25)：

| 参数 | 数量 | `parameters` 下标 |
| --- | ---: | --- |
| $W_1$ | $4\times2=8$ | $2h+i$，范围 $0\ldots7$ |
| $b_1$ | $4$ | $8+h$，范围 $8\ldots11$ |
| $W_2$ | $4$ | $12+h$，范围 $12\ldots15$ |
| $b_2$ | $1$ | $16$ |

总参数数为 $8+4+4+1=17$。这里没有动态内存分配；输入数组和梯度数组的容量由编译期常量固定。

## 2. 为什么单个神经元不能解决 XOR

单个 sigmoid 神经元的输出为 $p=\sigma(z)$，其中 $z=\alpha_0x_0+\alpha_1x_1+\beta$；$\alpha_0,\alpha_1$ 是两个输入权重，$\beta$ 是偏置，$\sigma$ 是 sigmoid 函数。按 $p\ge0.5$ 判为正类，等价于按 $z\ge0$ 分类，因此分类边界是一条直线。

两个正样本 $(0,1)$ 和 $(1,0)$ 要求：

$$
\alpha_1+\beta\ge0,\qquad \alpha_0+\beta\ge0.
$$

相加得到 $\alpha_0+\alpha_1+2\beta\ge0$。两个负样本 $(0,0)$ 和 $(1,1)$ 则要求：

$$
\beta<0,\qquad \alpha_0+\alpha_1+\beta<0.
$$

相加得到同一个左侧表达式 $\alpha_0+\alpha_1+2\beta<0$，与前一个不等式矛盾。增加训练轮数无法弥补模型表达能力不足；需要非线性隐藏层。源码在 [`network_forward`](../src/ml/06_xor_mlp.c#L99-L112) 中用 `tanh` 产生非线性特征，再由输出层组合。

## 3. 单个 sigmoid 神经元：从输入算到梯度

源码先用一个手算样例验证 sigmoid、损失和导数：$x_0=1$、$x_1=2$，权重 $w_0=0.5$、$w_1=-0.25$，偏置 $b=0.1$，标签 $y=1$。这些数值对应 [`test_neuron`](../src/ml/06_xor_mlp.c#L170-L193) 和 `main` 的演示输出。

### 前向计算与稳定交叉熵

logit 是线性分数，不是概率：

$$
z=w_0x_0+w_1x_1+b
=0.5\times1+(-0.25)\times2+0.1
=0.1.
$$

sigmoid 把实数分数映射到概率。$z\ge0$ 时用 $1/(1+e^{-z})$；$z<0$ 时源码改用 $e^z/(1+e^z)$，避免求可能溢出的大正指数：

$$
p=\sigma(z)=
\begin{cases}
\dfrac{1}{1+e^{-z}}, & z\ge0,\\[4pt]
\dfrac{e^z}{1+e^z}, & z<0.
\end{cases}
\qquad
p=\sigma(0.1)\approx0.5249791875.
$$

分支实现见 [`sigmoid`](../src/ml/06_xor_mlp.c#L40-L52)。浮点舍入可能让极端输入的概率恰好成为 $0$ 或 $1$，所以损失不能简单地对这个概率取对数。二分类交叉熵的常规定义是：

$$
L=-y\ln p-(1-y)\ln(1-p).
$$

其中 $y$ 是取 $0$ 或 $1$ 的标签，$\ln$ 是自然对数。将 sigmoid 代入后可改写成仅依赖 logit 的稳定形式：

$$
L(z,y)=\max(z,0)-yz+\ln\!\left(1+e^{-|z|}\right).
$$

对本例，$y=1$ 且 $z=0.1$：

$$
L=0.1-1\times0.1+\ln(1+e^{-0.1})
=\ln(1+0.9048374)
\approx0.6443966601.
$$

源码使用 `fmax`、`fabs` 和 `log1p` 实现此式，见 [`binary_cross_entropy`](../src/ml/06_xor_mlp.c#L54-L63)。例如 $z=1000,y=0$ 时，损失为 $1000+\ln(1+e^{-1000})\approx1000$，不会先把 $p$ 舍入为 $1$ 再计算 $\ln(1-p)$。

### 从交叉熵推导输出梯度

先对概率求导，再乘 sigmoid 的导数。利用 $dp/dz=p(1-p)$：

$$
\frac{dL}{dz}
=\left(-\frac{y}{p}+\frac{1-y}{1-p}\right)p(1-p)
=-y(1-p)+(1-y)p
=p-y.
$$

本例代入 $p\approx0.5249791875$、$y=1$：

$$
\delta_{\mathrm{out}}=\frac{dL}{dz}=p-y
=0.5249791875-1
\approx-0.4750208125.
$$

这里 $\delta_{\mathrm{out}}$ 表示输出 logit 的梯度。权重通过 $z=w_0x_0+w_1x_1+b$ 影响损失，因此：

$$
\frac{\partial L}{\partial w_0}
=\delta_{\mathrm{out}}x_0
=-0.4750208125\times1
=-0.4750208125,
$$

$$
\frac{\partial L}{\partial w_1}
=\delta_{\mathrm{out}}x_1
=-0.4750208125\times2
=-0.9500416250,
\qquad
\frac{\partial L}{\partial b}
=\delta_{\mathrm{out}}
\approx-0.4750208125.
$$

因此实现中不再额外乘 $p(1-p)$：这个因子已经在 sigmoid 与交叉熵的合并求导中抵消。`test_neuron` 用中心差分对 $dL/dz$ 做独立数值核对。

## 4. 初始化如何变成确定的参数

[`random_weight`](../src/ml/06_xor_mlp.c#L65-L75) 使用 `uint32_t` 状态，而不是平台实现可能不同的 `rand()`。令 $q_k$ 为第 $k$ 次更新前的 32 位无符号状态，$r_k$ 为更新后状态右移 $8$ 位的整数，$u_k$ 为归一化值，则源码对应：

$$
q_{k+1}=(1664525q_k+1013904223)\bmod2^{32},\qquad
r_k=\left\lfloor\frac{q_{k+1}}{2^8}\right\rfloor,
$$

$$
u_k=\frac{r_k}{2^{24}}\in[0,1),\qquad
w_k=1.6u_k-0.8\in[-0.8,0.8).
$$

这里 $q_0=42$ 是本例种子，$k$ 是权重生成序号；模 $2^{32}$ 对应无符号整数回绕。`network_initialize` 先把整个网络清零，再对每个隐藏神经元依次生成两个 $W_1$ 权重和一个 $W_2$ 权重，所以所有偏置初始为零。

按此算法从种子 $42$ 生成、四舍五入到小数点后六位的初始参数为：

$$
W_1=\begin{bmatrix}
-0.396248 & -0.659000\\
 -0.443913 & -0.198944\\
-0.084350 & -0.610464\\
 0.791415 &  0.565124
\end{bmatrix},\qquad
W_2=\begin{bmatrix}
0.123650 & -0.758938 & 0.598102 & -0.000517
\end{bmatrix},
$$

$$
b_1=\begin{bmatrix}0&0&0&0\end{bmatrix}^{\mathsf T},\qquad b_2=0.
$$

矩阵行对应 $h=0,1,2,3$，列对应 $i=0,1$。这些数值是按源码的整数生成步骤计算后显示的近似值；后续手算使用未舍入的权重，只在文档中显示有限小数位。相同编译环境中的两次初始化会逐位相等，源码也逐项断言这一点。不同数学库实现可能令后续 `tanh` 结果有细微舍入差异。

## 5. 真实 XOR 网络的一次完整前向与更新

训练循环中的第一个样本是 $n=0$：$x_0=(0,0)$、$y_0=0$。令 $u_h$ 为隐藏神经元的加权输入，$a_h$ 为 `tanh` 输出。前向传播公式与 [`network_forward`](../src/ml/06_xor_mlp.c#L99-L112) 一一对应：

$$
u_h=b_{1,h}+\sum_{i=0}^{1}W_{1,h,i}x_{0,i},\qquad
a_h=\tanh(u_h),
$$

$$
z=b_2+\sum_{h=0}^{3}W_{2,h}a_h,\qquad p=\sigma(z).
$$

源码在每个隐藏单元中累加输入，再把 `tanh` 结果立刻送入输出层；末尾才计算 sigmoid：

```c
double preactivation = network->parameters[B1 + h];
for (size_t i = 0; i < INPUTS; ++i) {
    preactivation += network->parameters[W1 + h * INPUTS + i] * x[i];
}
forward.hidden[h] = tanh(preactivation);
forward.logit += network->parameters[W2 + h] * forward.hidden[h];
```

对应 [`network_forward`](../src/ml/06_xor_mlp.c#L99-L112)：`preactivation` 对应 $u_h$，`forward.hidden[h]` 对应 $a_h$，`forward.logit` 对应 $z$。循环结束后第 $111$ 行才计算 `forward.probability = sigmoid(forward.logit)`。

由于两个输入均为零、初始隐藏偏置也为零，对每个 $h$ 都有：

$$
u_h=0+W_{1,h,0}\times0+W_{1,h,1}\times0=0,
\qquad a_h=\tanh(0)=0.
$$

所以 $z=0+\sum_{h=0}^{3}W_{2,h}\times0=0$，$p=\sigma(0)=0.5$。标签 $y_0=0$，该样本损失为：

$$
L_0=\max(0,0)-0\times0+\ln(1+e^0)=\ln2\approx0.6931471806.
$$

反向传播先得 $\delta_{\mathrm{out}}=p-y_0=0.5$。对隐藏层，令 $\delta_h$ 表示损失对 $u_h$ 的导数。因为 $d\tanh(u_h)/du_h=1-a_h^2$，链式法则给出：

$$
\delta_h=\delta_{\mathrm{out}}W_{2,h}(1-a_h^2),\qquad
\frac{\partial L}{\partial W_{2,h}}=\delta_{\mathrm{out}}a_h,\qquad
\frac{\partial L}{\partial b_2}=\delta_{\mathrm{out}},
$$

$$
\frac{\partial L}{\partial W_{1,h,i}}=\delta_hx_{0,i},\qquad
\frac{\partial L}{\partial b_{1,h}}=\delta_h.
$$

此时 $a_h=0$，所以输出权重梯度全为零；$x_{0,i}=0$，所以输入权重梯度全为零。隐藏偏置梯度则为 $\delta_h=0.5W_{2,h}$。代入初始 $W_2$ 得：

$$
\nabla_{b_1}L_0\approx
\begin{bmatrix}
0.061824942 & -0.379468918 & 0.299050951 & -0.000258636
\end{bmatrix}^{\mathsf T},
\qquad
\nabla_{b_2}L_0=0.5.
$$

`network_backward` 将这些梯度写入数组；它不会修改网络。接着训练率 $\eta=0.1$，`network_update` 对每个参数下标 $r\in\{0,\ldots,16\}$ 执行 $\theta_r\leftarrow\theta_r-\eta g_r$，其中 $\theta_r$ 是参数，$g_r$ 是对应梯度。于是这一步后：

$$
b_1\approx
\begin{bmatrix}
-0.006182494 & 0.037946892 & -0.029905095 & 0.000025864
\end{bmatrix}^{\mathsf T},\qquad b_2=-0.05,
$$

而 $W_1$、$W_2$ 保持原值。这展示了为什么“输入为零”时输入权重没有更新，但隐藏偏置和输出偏置仍能更新。反向传播必须读取更新前的 $W_2$；因此代码先完整计算梯度，再调用 [`network_update`](../src/ml/06_xor_mlp.c#L140-L153)。

紧接着循环处理 $n=1$，即 $x_1=(0,1)$、$y_1=1$。使用刚更新的偏置，四个隐藏层加权输入约为：

$$
u\approx\begin{bmatrix}-0.665183&-0.160997&-0.640369&0.565150\end{bmatrix}^{\mathsf T},
\qquad
a=\tanh(u)\approx\begin{bmatrix}-0.581802&-0.159620&-0.565151&0.511789\end{bmatrix}^{\mathsf T}.
$$

输出层将四个特征加权求和：

$$
z\approx-0.05-0.071940+0.121142-0.338018-0.000265=-0.339081,
\qquad
p=\sigma(z)\approx0.416033.
$$

因为正确标签是 $1$，这次概率仍偏低，稳定交叉熵约为 $L_1=0.876991$。下一次反向传播会产生非零的 $W_1$ 梯度，因为输入的第二个分量为 $1$；随后更新会同时改变隐藏特征和输出。这里展示的是源码固定顺序 SGD 的前两次样本处理，不是独立于训练循环的虚构批次。

## 6. 隐藏层反向传播逐项解释

把上面的链式法则推广到任意输入。对每个隐藏下标 $h$ 和输入下标 $i$：

$$
u_h=b_{1,h}+\sum_{i=0}^{1}W_{1,h,i}x_i,
\qquad a_h=\tanh(u_h),
\qquad z=b_2+\sum_{h=0}^{3}W_{2,h}a_h.
$$

令 $y$ 是当前样本标签，$p=\sigma(z)$ 是前向缓存的概率。输出层误差信号为 $\delta_{\mathrm{out}}=p-y$；继续向隐藏层传递时乘以输出权重和 `tanh` 的导数：

$$
\delta_h=\frac{\partial L}{\partial u_h}
=\delta_{\mathrm{out}}W_{2,h}(1-a_h^2).
$$

例如当某隐藏激活 $a_h=0.5$、对应输出权重 $W_{2,h}=0.4$、输出误差 $\delta_{\mathrm{out}}=-0.2$ 时：

$$
\delta_h=(-0.2)\times0.4\times(1-0.5^2)
=-0.08\times0.75=-0.06.
$$

这个隐藏神经元的输入权重梯度等于其误差信号乘以输入。若输入分量 $x_i=2$，则 $\partial L/\partial W_{1,h,i}=-0.06\times2=-0.12$；偏置梯度为 $\partial L/\partial b_{1,h}=-0.06$。这些例子是对源码公式的手算示范，数值不声称是训练过程中的某个隐藏单元状态。

代码中 `forward.hidden[h]` 保存 $a_h$，所以无需再次调用 `tanh`。`network_backward` 中的 `network->parameters[W2 + h]` 是刚才前向传播对应的旧输出权重；若先更新它再计算 $\delta_h$，梯度就不再对应同一个参数版本。

源码的关键计算顺序如下：

```c
double output_delta = forward->probability - target;
gradients[W2 + h] = output_delta * forward->hidden[h];
double hidden_delta = output_delta * network->parameters[W2 + h] *
                      (1 - forward->hidden[h] * forward->hidden[h]);
gradients[B1 + h] = hidden_delta;
```

这几行分别对应 $\delta_{\mathrm{out}}$、$\partial L/\partial W_{2,h}$、$\delta_h$ 和 $\partial L/\partial b_{1,h}$。随后内层循环将 `hidden_delta * x[i]` 写入 $W_1$ 对应的梯度槽位。

## 7. SGD、平均损失与训练输出

一个 epoch 是按固定顺序访问四个样本一次。每个样本都独立执行前向、反向、更新，因此这里是在线 SGD，而不是整批梯度下降：

$$
\theta^{(t+1)}=\theta^{(t)}-\eta\nabla_\theta L_n(\theta^{(t)}),
\qquad \eta=0.1.
$$

其中 $t$ 是单次样本更新序号，$n$ 是当前样本序号，$\theta$ 是全部 $17$ 个参数组成的向量。源码每轮有四次更新，总共执行 $5000\times4=20000$ 次更新。输出的平均损失是在更新后用同一份当前网络重新计算：

```c
Forward forward = network_forward(&network, xor_inputs[sample]);
double gradients[PARAMETERS] = {0};
network_backward(&network, xor_inputs[sample], xor_targets[sample], &forward, gradients);
network_update(&network, gradients, LEARNING_RATE);
```

这段固定顺序循环位于 [`main`](../src/ml/06_xor_mlp.c#L288-L296)。每次 `network_update` 对参数数组逐项执行：

```c
network->parameters[p] -= learning_rate * gradients[p];
```

也就是 $\theta_r\leftarrow\theta_r-\eta g_r$；实现同时断言更新后的每个参数仍是有限数。

$$
J(\theta)=\frac{1}{4}\sum_{n=0}^{3}L(z_n,y_n).
$$

这里 $z_n$ 是当前网络对第 $n$ 个样本算出的 logit，$y_n$ 是对应标签。用种子 $42$ 的初始参数计算，四个样本损失依次约为 $0.693147$、$0.824990$、$0.589571$ 和 $0.679898$，因此：

$$
J(\theta^{(0)})\approx\frac{0.693147+0.824990+0.589571+0.679898}{4}
\approx0.696902.
$$

固定环境本次运行的训练观察值为：

```text
epoch=   0，平均交叉熵=0.696902
epoch= 500，平均交叉熵=0.023407
epoch=1000，平均交叉熵=0.008552
epoch=1500，平均交叉熵=0.005101
epoch=2000，平均交叉熵=0.003601
epoch=2500，平均交叉熵=0.002770
epoch=3000，平均交叉熵=0.002245
epoch=3500，平均交叉熵=0.001885
epoch=4000，平均交叉熵=0.001623
epoch=4500，平均交叉熵=0.001425
epoch=5000，平均交叉熵=0.001269
```

程序按 $p\ge0.5$ 判为 $1$，最终输出为：

```text
x=[0,0]，目标=0，概率=0.000234，预测=0
x=[0,1]，目标=1，概率=0.998497，预测=1
x=[1,0]，目标=1，概率=0.998746，预测=1
x=[1,1]，目标=0，概率=0.002081，预测=0
```

源码断言最终损失有限、低于初始损失且小于 $0.02$；正样本概率大于 $0.95$，负样本概率小于 $0.05$。这些检查证明本次实现拟合了完整布尔真值表，不代表模型能泛化到其他数据，也不保证任意种子或学习率都有相同曲线。

## 8. 为什么初始化不能全部为零

全零初始化时，每个隐藏层加权输入和激活都是零，输出概率为 $0.5$。对任一 XOR 样本，输出权重梯度为 $\delta_{\mathrm{out}}a_h=0$；隐藏误差信号也含有 $W_{2,h}=0$，所以 $W_1$ 和 $b_1$ 的梯度为零。每个隐藏单元得到完全相同的信号，无法学出不同特征。

此外 XOR 标签中有两个 $0$ 和两个 $1$。对输出偏置，四个样本的梯度分别为 $0.5,-0.5,-0.5,0.5$，平均值为：

$$
\frac{0.5-0.5-0.5+0.5}{4}=0.
$$

因此源码测试中全零网络的四样本平均梯度全部为零，网络会停在 $p=0.5$。仅把偏置设为零并没有这个问题：随机权重已经让不同隐藏单元得到不同输入响应。固定种子使实验可复现，但不是每个随机初始化都能保证同样快地收敛。

## 9. 梯度检查与测试实际证明什么

反向传播公式写错时，程序仍可能训练出看似合理的结果，因此源码另用有限差分检查解析梯度。令 $\theta_r$ 是第 $r$ 个参数，$\varepsilon=10^{-5}$ 是扰动大小，$\mathbf e_r$ 是仅第 $r$ 项为 $1$ 的单位向量，则数值梯度为：

$$
g^{\mathrm{num}}_r=
\frac{L(\theta+\varepsilon\mathbf e_r)-L(\theta-\varepsilon\mathbf e_r)}{2\varepsilon}.
$$

分子是在只增加或减少一个参数后重新前向计算的损失差；除以 $2\varepsilon$ 是中心差分近似。源码对四个样本、每个样本的 $17$ 个参数都进行比较。接受条件为：

$$
|g^{\mathrm{num}}_r-g^{\mathrm{ana}}_r|
<10^{-8}+10^{-6}\max\!\left(|g^{\mathrm{num}}_r|,|g^{\mathrm{ana}}_r|\right),
$$

其中 $g^{\mathrm{ana}}_r$ 是 `network_backward` 写出的解析梯度。绝对误差项照顾接近零的梯度，相对误差项随梯度大小缩放。本次 `make test-xor-mlp` 输出最大绝对误差 $1.399\times10^{-11}$。

测试还覆盖：极端 logit 下 sigmoid 和 BCE 的数值行为；种子 $42$ 重复初始化完全相同；一次 $\eta=0.01$ 更新符合参数减梯度的公式且降低被测样本损失；训练结束后参数和概率有限，四个预测正确。消毒器通过说明本次路径没有发现地址或未定义行为问题，但不证明任意输入、编译器或平台都没有缺陷。

## 10. C 实现与学习边界

- `Network` 和 `Forward` 使用固定长度数组，内存布局简单，没有堆分配或外部深度学习框架。
- `uint32_t`、`size_t`、`assert` 和数学函数来自标准头文件；构建命令通过 `-lm` 链接数学库。不同编译器对 C23 的支持程度可能不同，但本案例没有依赖 C23 独有语法。
- `assert` 用于教学验证；若定义 `NDEBUG`，断言会被移除，因此验证时应保持断言开启。
- 输入、标签和参数的合法范围由本固定案例保证；网络函数不是带完整运行时校验的通用库接口。
- 数据顺序固定，没有打乱、小批量、Adam、模型保存或真实数据评估；这些属于后续阶段主题。

## 练习

1. 在 [`network_forward`](../src/ml/06_xor_mlp.c#L99-L113) 断点观察 $u_h$、$a_h$、$z$ 和 $p$，用上面的公式手算同一个样本。
2. 对某个 $h$，分别解释输出梯度 $\partial L/\partial W_{2,h}$、隐藏误差信号 $\delta_h$ 和输入权重梯度 $\partial L/\partial W_{1,h,i}$ 的来源。
3. 将网络参数全部清零，预测四个输入的概率，并对照对称性推导。
4. 保留有限差分检查，改变隐藏宽度、学习率或样本顺序，观察哪些断言和训练曲线会变化。
5. 证明增加任意多个纯线性层仍只能得到一个线性变换；再思考 `tanh` 为何能让隐藏层产生非线性特征。

理解了前向、梯度和更新之间的对应关系后，再进入阶段 2 的二维分类、决策边界和更复杂训练策略。
