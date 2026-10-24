# 纯 C GAN：从八簇二维点理解对抗训练

本文结合 [`src/ml/14_gan_2d.c`](../src/ml/14_gan_2d.c) 说明一个不依赖机器学习框架的生成对抗网络。模型用两个手写 MLP 学习圆周上的八簇二维点：生成器把随机噪声变成坐标，判别器判断坐标来自真实分布还是生成器。文中公式对应当前源码，不代表所有 GAN 变体都采用相同损失或训练顺序。

![GAN 生成器和判别器交替更新的流程](images/60_gan_2d.svg)

生成器把噪声映射到二维点，判别器区分真实点与生成点；更新生成器时梯度经过判别器，但该步不更新判别器参数。

## 运行与观察

```sh
make gan                              # 运行验证、训练并导出四张 BMP
make test-gan                         # 执行数值检查和完整训练，不写文件
make sanitize-gan                     # 用地址与未定义行为消毒器运行测试
./build/ml/14_gan_2d                   # 训练并显示报告
./build/ml/14_gan_2d --export --log     # 导出图片，并双输出训练日志
less logs/ml/14_gan_2d.log
```

默认训练预算是 $6000$ 步，每步批量大小为 $32$。`--test` 仍会运行梯度断言和完整训练，只关闭中间进度及文件输出；不能与 `--export` 或 `--log` 同用。图片独占创建于 `exports/`，日志追加至 `logs/ml/14_gan_2d.log`。实现已接入默认构建、测试、严格检查、调试程序和编译数据库；模型计算只依赖 ISO C 与数学库，图像导出使用项目的纯 C 画布和 BMP 接口。

## 一、这个 GAN 要学什么

### 真实分布和潜在噪声

共有 $8$ 个模式，索引 $k$ 从 $0$ 到 $7$。第 $k$ 个模式的中心 $\mathbf{c}_k$ 位于半径为 $2$ 的圆周上：

$$
\mathbf{c}_k=2\begin{bmatrix}
\cos(2\pi k/8)\\
\sin(2\pi k/8)
\end{bmatrix},
\qquad
\mathbf{x}=\mathbf{c}_k+0.08\boldsymbol{\epsilon},
\quad
k\sim\operatorname{Uniform}\{0,\ldots,7\},
\quad
\boldsymbol{\epsilon}\sim\mathcal{N}(\mathbf{0},I_2).
$$

这里 $\mathbf{x}\in\mathbb{R}^2$ 是无量纲真实坐标，$I_2$ 是二维单位矩阵；模式索引 $k$ 与噪声向量 $\boldsymbol{\epsilon}$ 独立，噪声的两个分量也相互独立且服从标准正态分布。系数 $0.08$ 是每个坐标方向的标准差，所以每个模式是一个小而圆的二维高斯簇。`sample_real` 每次都重新抽取模式和噪声，没有固定训练文件或逐点标签需要记忆，见 [`mode_center` 与 `sample_real`](../src/ml/14_gan_2d.c#L112-L133)。

生成器的输入也有两个独立标准正态分量：

$$
\mathbf{z}\sim\mathcal{N}(\mathbf{0},I_2),
\qquad
\mathbf{y}=G_{\theta}(\mathbf{z})\in\mathbb{R}^2.
$$

$\mathbf{z}$ 是潜在噪声，$\mathbf{y}$ 是生成坐标，$\theta$ 表示生成器参数。生成器不接收真实点作为配对目标；不同噪声可以映射到同一模式，训练目标是让整体生成分布接近真实分布。

### 随机数如何变成二维样本

`random_unit` 使用 `uint32_t` 线性同余递推。无符号整数溢出按模 $2^{32}$ 回绕；取高 $24$ 位并加半个单位，使均匀数不等于 $0$ 或 $1$：

$$
s_{n+1}=(1664525s_n+1013904223)\bmod 2^{32},
\qquad
U_n=\frac{\lfloor s_{n+1}/2^8\rfloor+0.5}{2^{24}}.
$$

$s_n$ 是伪随机状态，$U_n\in(0,1)$。`sample_normal` 用 Box–Muller 将两个均匀数 $U_1,U_2$ 变成两个标准正态数：

$$
r=\sqrt{-2\ln U_1},
\qquad
\vartheta=2\pi U_2,
\qquad
\epsilon_0=r\cos\vartheta,
\quad
\epsilon_1=r\sin\vartheta.
$$

例如真实数据随机状态从 $s_0=2025$ 开始，源码首轮抽到模式选择数 $U=0.020861655474$，因此 $k=\lfloor8U\rfloor=0$。接着 $U_1=0.953691393137$、$U_2=0.353785246611$，得到 $\boldsymbol{\epsilon}\approx(-0.186879174209,0.244757979516)$。因为 $\mathbf{c}_0=(2,0)$，第一个真实点是

$$
\mathbf{x}=(2,0)+0.08(-0.186879174209,0.244757979516)
\approx(1.985049666063,0.019580638361).
$$

这些值对应 `sample_real` 和 `sample_normal` 的实际伪随机序列；随机数只为教学实验提供可复现输入，不适用于安全用途。真实样本流、判别器噪声流和生成器噪声流各用独立状态；Box–Muller 在同一个状态中连续取两个均匀数。

## 二、源码地图与参数布局

| 概念 | 源码位置 | 关键数据 | 作用 |
| --- | --- | --- | --- |
| 网络和前向缓存 | [`Network`、`Forward`、偏移宏](../src/ml/14_gan_2d.c#L13-L57) | `parameters`、`input`、`first`、`second`、`output` | 固定数组保存参数和中间激活 |
| 初始化与前向 | [`network_initialize`](../src/ml/14_gan_2d.c#L141-L154)、[`network_forward`](../src/ml/14_gan_2d.c#L162-L188) | `W1` 到 `B3` | Xavier 初始化并逐层计算两个隐藏层和线性输出 |
| 反向传播 | [`network_backward`](../src/ml/14_gan_2d.c#L198-L229) | `upstream`、`second`、`first`、`input_gradient` | 累加参数梯度，并计算传给前一网络的输入梯度 |
| 判别损失和生成器链 | [`binary_loss`](../src/ml/14_gan_2d.c#L236-L254)、[`generator_backward`](../src/ml/14_gan_2d.c#L265-L279) | logit、BCE、`point_gradient` | 稳定计算损失，并通过冻结的判别器把梯度传给生成器 |
| 更新和训练 | [`adam_update`](../src/ml/14_gan_2d.c#L288-L319)、[`train`](../src/ml/14_gan_2d.c#L375-L435) | 两套 Adam、`dg`、`gg` | 先更新判别器，再用新的判别器更新生成器 |
| 测试与评估 | [`test_gradients`、`test_boundaries`](../src/ml/14_gan_2d.c#L441-L541)、[`evaluate`](../src/ml/14_gan_2d.c#L329-L365) | 中心差分、簇距离、覆盖数 | 检查梯度实现并量化已知合成分布上的生成结果 |
| 图片与入口 | [`export_images`](../src/ml/14_gan_2d.c#L611-L649)、[`main`](../src/ml/14_gan_2d.c#L657-L722) | `--export`、`--log`、`--test` | 训练、报告、可选导出和错误处理 |

`Network.parameters` 是长度为 $354$ 的 `double` 数组。权重按 `[输出][输入]` 行优先排列，宏计算出的偏移如下：

| C 偏移宏 | 张量 | 生成器形状 | 判别器形状 |
| --- | --- | ---: | ---: |
| `W1 = 0` | $W^{(1)}$ | $16\times2$ | $16\times2$ |
| `B1 = 32` | $\mathbf{b}^{(1)}$ | $16$ | $16$ |
| `W2 = 48` | $W^{(2)}$ | $16\times16$ | $16\times16$ |
| `B2 = 304` | $\mathbf{b}^{(2)}$ | $16$ | $16$ |
| `W3 = 320` | $W^{(3)}$ | $2\times16$ | $1\times16$ |
| `B3 = 352` | $\mathbf{b}^{(3)}$ | $2$ | $1$ |

因此生成器有 $32+16+256+16+32+2=354$ 个有效参数，判别器有 $32+16+256+16+16+1=337$ 个有效参数。为共用固定容量结构，判别器也占用 $354$ 个数组槽；末层第二行和第二个偏置未参与计算，保持为零。模型和缓存都由调用方持有，没有堆分配。`network_initialize` 先将网络整体清零，再按实际输出数初始化权重；偏置因此为零。

## 三、两个 MLP 的前向计算

生成器的结构是 $2\to16(\tanh)\to16(\tanh)\to2$，判别器是 $2\to16(\tanh)\to16(\tanh)\to1$；两者最后一层都是线性层。对任一网络输入 $\mathbf{q}$，前向计算可写为

$$
\mathbf{a}^{(1)}=W^{(1)}\mathbf{q}+\mathbf{b}^{(1)},
\qquad
\mathbf{h}^{(1)}=\tanh(\mathbf{a}^{(1)}),
$$

$$
\mathbf{a}^{(2)}=W^{(2)}\mathbf{h}^{(1)}+\mathbf{b}^{(2)},
\qquad
\mathbf{h}^{(2)}=\tanh(\mathbf{a}^{(2)}),
\qquad
\mathbf{o}=W^{(3)}\mathbf{h}^{(2)}+\mathbf{b}^{(3)}.
$$

上标 $(1),(2),(3)$ 表示层号，不是乘方；$\mathbf{a}$ 是激活函数前的值，$\mathbf{h}$ 是隐藏层激活，$\mathbf{o}$ 是线性输出。生成器的 $\mathbf{q}=\mathbf{z}$ 且 $\mathbf{o}=\mathbf{y}$ 有两个坐标；判别器的 $\mathbf{q}=\mathbf{x}$ 或 $\mathbf{y}$，输出只有一个 logit $a_D$。`network_forward` 将隐藏值保存在 `Forward.first`、`Forward.second`，线性值保存在 `Forward.output`，便于反向重用。矩阵元素通过 `W1 + i * inputs + j` 这样的索引读取。

权重按 Xavier 均匀分布初始化。每层输入维数为 $n_{\mathrm{in}}$、输出维数为 $n_{\mathrm{out}}$ 时，源码设置

$$
W_{ij}\sim\operatorname{Uniform}(-\alpha_{\mathrm{init}},\alpha_{\mathrm{init}}),
\qquad
\alpha_{\mathrm{init}}=\sqrt{\frac{6}{n_{\mathrm{in}}+n_{\mathrm{out}}}}.
$$

这里 $i$、$j$ 分别索引输出单元和输入单元；两者从 $0$ 开始，分别小于本层输出维数和输入维数。`random_unit` 产生 $U\in(0,1)$ 后，代码用 $\alpha_{\mathrm{init}}(2U-1)$ 映射到该范围；C 局部变量 `scale` 保存 $\alpha_{\mathrm{init}}$，偏置不随机。初始化种子分别为生成器 $42$ 和判别器 $43$。以生成器第一层为例，$n_{\mathrm{in}}=2$、$n_{\mathrm{out}}=16$，所以 $\alpha_{\mathrm{init}}=\sqrt{6/18}\approx0.577350269$；固定种子下首个权重为 $-0.285967160278$，确实落在初始化范围内。连续均匀分布是数学概括；实际初始化值由 `random_unit` 的 $24$ 位网格离散化。

### 用固定输入手算生成器前向

`test_gradients` 将两个网络分别用种子 $42$、$43$ 初始化，并使用 $\mathbf{z}=(0.7,-0.4)$ 检查完整梯度链。这个输入是有限差分测试夹具，不是训练时必然抽到的噪声。对生成器第一层的第 $0$ 个单元，实际权重为 $W^{(1)}_{0,0}=-0.285967160278$、$W^{(1)}_{0,1}=-0.475592260844$，偏置为 $0$：

$$
a^{(1)}_0=(-0.285967160278)(0.7)+(-0.475592260844)(-0.4)+0
=-0.200177012195+0.190236904338
=-0.009940107857,
$$

$$
h^{(1)}_0=\tanh(-0.009940107857)=-0.009939780490.
$$

完整 $16$ 维隐藏层继续计算后，第二层第 $0$ 个单元为 $a^{(2)}_0=-0.025791219140$、$h^{(2)}_0=-0.025785502000$。输出层再把全部 $16$ 个第二层激活乘以对应权重并求和，得到

$$
\mathbf{y}=G(0.7,-0.4)=(-0.154483973954,0.107332626517).
$$

这个算例对应代码的两层 `tanh` 循环和线性输出循环。它也说明生成坐标没有 sigmoid 或裁切，可能落在训练簇附近、簇间区域，甚至绘图范围以外。

## 四、判别器概率与稳定二元交叉熵

判别器最后输出实数 logit $a$，再用 sigmoid 转成真实类别概率 $p$：

$$
\sigma(a)=
\begin{cases}
\dfrac{1}{1+e^{-a}}, & a\ge 0,\\
\dfrac{e^a}{1+e^a}, & a<0,
\end{cases}
\qquad
p=D(\mathbf{x})=\sigma(a_D(\mathbf{x})).
$$

符号 $D(\mathbf{x})$ 表示判别器认为输入是真实点的概率，并不是数据分布的概率密度。按符号切换等价形式，可避免对很大的正数计算 $e^{|a|}$；实现见 [`sigmoid`](../src/ml/14_gan_2d.c#L236-L243)。

标签 $y=1$ 表示真实点，$y=0$ 表示生成点。logit 形式的二元交叉熵为

$$
\ell(a,y)=-y\ln\sigma(a)-(1-y)\ln(1-\sigma(a))
=\operatorname{softplus}(a)-ya.
$$

其中 $a$ 是未经过 sigmoid 的分数，$y\in\{0,1\}$ 是目标标签，$\operatorname{softplus}(a)=\ln(1+e^a)$。当 $y=1$，损失为 $-\ln\sigma(a)$；当 $y=0$，损失为 $-\ln(1-\sigma(a))$。直接计算这两个概率对极端值可能遇到 `log(0)`，所以代码用等价稳定式：

$$
\ell(a,y)=\max(a,0)-ya+\ln(1+e^{-|a|}).
$$

源码对应 `fmax(logit, 0) - target * logit + log1p(exp(-fabs(logit)))`；`log1p(v)` 稳定计算 $\ln(1+v)$。对 logit 求导时，sigmoid 导数与概率分母抵消：

$$
\begin{aligned}
\frac{\partial\ell}{\partial a}
&=\left(-\frac{y}{p}+\frac{1-y}{1-p}\right)p(1-p)\\
&=-y(1-p)+(1-y)p\\
&=p-y=\sigma(a)-y.
\end{aligned}
$$

这就是 `train` 里 `delta = (sigmoid(logit) - target) / BATCH_SIZE` 的来源。若真实样本的 $p=0.514179609378$，则 $y=1$ 时 $\partial\ell/\partial a=-0.485820390622$，梯度下降会增大 logit；若同一个点被当作假样本，$y=0$ 时导数为 $0.514179609378$，梯度下降会减小 logit。

## 五、判别器与生成器优化的目标不同

经典 GAN 的价值函数写作

$$
\min_G\max_D V(D,G)
=\mathbb{E}_{\mathbf{x}\sim p_{\mathrm{data}}}[\ln D(\mathbf{x})]
+\mathbb{E}_{\mathbf{z}\sim p_{\mathbf{z}}}[\ln(1-D(G(\mathbf{z})))].
$$

$p_{\mathrm{data}}$ 是上文的八簇真实分布，$p_{\mathbf{z}}$ 是标准二维正态分布。判别器最大化此值，等价于最小化真实和生成两类的 BCE 之和。源码每步使用批量 $B=32$，目标是

$$
\mathcal{L}_D=
\frac{1}{B}\sum_{b=1}^{B}\ell(a_D(\mathbf{x}_b),1)
+\frac{1}{B}\sum_{b=1}^{B}\ell(a_D(G(\mathbf{z}_b)),0).
$$

$b$ 是当前批次中的样本索引，$\mathbf{x}_b$ 是实时抽样的真实点，$\mathbf{z}_b$ 是判别器阶段使用的噪声。注意源码将两组均值相加，没有再除以 $2$；当真实和生成概率都恰为 $0.5$ 时，参考判别损失为 $2\ln2\approx1.386294$。

若严格使用原始极小极大目标，生成器会最小化 $\ln(1-D(G(\mathbf{z})))$。本案例使用更常见的非饱和目标：

$$
\mathcal{L}_G=\frac{1}{B}\sum_{b=1}^{B}\ell(a_D(G(\tilde{\mathbf{z}}_b)),1)
=-\frac{1}{B}\sum_{b=1}^{B}\ln D(G(\tilde{\mathbf{z}}_b)).
$$

$\tilde{\mathbf{z}}_b$ 是生成器阶段重新抽取的噪声，不与判别器阶段的 $\mathbf{z}_b$ 配对。损失目标 $1$ 表示生成器要让判别器把生成点判真；这不是把真实数据当成生成器的逐点答案。

两种生成器目标的 logit 梯度不同：

$$
\frac{\partial\ln(1-\sigma(a))}{\partial a}=-\sigma(a),
\qquad
\frac{\partial[-\ln\sigma(a)]}{\partial a}=\sigma(a)-1.
$$

若判别器对假样本很确信，$a\ll0$、$\sigma(a)\approx0$，原始目标的导数 $-\sigma(a)$ 接近 $0$，生成器学习信号容易变弱；非饱和目标导数接近 $-1$，仍推动 logit 增大。代码的 `binary_loss(logit, 1)` 正是 $-\ln\sigma(a)$ 的稳定形式。判别器在自己的更新阶段把假样本标为 $0$，而生成器更新阶段把同一类输出的期望方向设为 $1$；这是对抗双方不同的优化目标。

在 $D(\mathbf{x})=D(G(\mathbf{z}))=0.5$ 时，参考生成器损失为 $\ln2\approx0.693147$。$\mathcal{L}_D$ 和 $\mathcal{L}_G$ 会彼此牵动，不要求每一步都下降；损失值本身或 $D$ 接近 $0.5$ 都不能单独证明模式覆盖良好。

## 六、反向传播如何沿网络回传

对任意一个网络，令 $\boldsymbol{\delta}^{(3)}=\partial\mathcal{L}/\partial\mathbf{o}$ 为输出梯度。权重梯度是当前层误差与输入激活的外积：

$$
\begin{aligned}
\nabla_{W^{(3)}}\mathcal{L}&=\boldsymbol{\delta}^{(3)}(\mathbf{h}^{(2)})^{\mathsf T},\\
\nabla_{\mathbf{b}^{(3)}}\mathcal{L}&=\boldsymbol{\delta}^{(3)},\\
\mathbf{r}^{(2)}&=(W^{(3)})^{\mathsf T}\boldsymbol{\delta}^{(3)},\\
\boldsymbol{\delta}^{(2)}&=\mathbf{r}^{(2)}\odot\left(\mathbf{1}-(\mathbf{h}^{(2)})^2\right),\\
\nabla_{W^{(2)}}\mathcal{L}&=\boldsymbol{\delta}^{(2)}(\mathbf{h}^{(1)})^{\mathsf T},\\
\mathbf{r}^{(1)}&=(W^{(2)})^{\mathsf T}\boldsymbol{\delta}^{(2)},\\
\boldsymbol{\delta}^{(1)}&=\mathbf{r}^{(1)}\odot\left(\mathbf{1}-(\mathbf{h}^{(1)})^2\right),\\
\nabla_{W^{(1)}}\mathcal{L}&=\boldsymbol{\delta}^{(1)}\mathbf{q}^{\mathsf T},\\
\frac{\partial\mathcal{L}}{\partial\mathbf{q}}&=(W^{(1)})^{\mathsf T}\boldsymbol{\delta}^{(1)}.
\end{aligned}
$$

这里 $\odot$ 表示逐元素乘法，$\mathbf{1}$ 是与隐藏向量同长度的全 $1$ 向量，$\mathsf T$ 表示矩阵转置；平方也逐元素计算。偏置梯度等于误差向量。`network_backward` 的 `upstream` 是 $\boldsymbol{\delta}^{(3)}$，局部数组 `second`、`first` 依次保存 $\mathbf{r}^{(2)}$、$\mathbf{r}^{(1)}$，局部变量 `delta` 保存对应的隐藏层误差。参数梯度使用 `+=` 累加，`input_gradient` 每次先清零，再覆盖成网络输入的梯度。

### 生成器为什么需要经过冻结的判别器反传

固定生成器输出 $\mathbf{y}=G_{\theta}(\mathbf{z})$，判别器给出 logit $a_D(\mathbf{y})$。生成器损失对判别 logit 的梯度是

$$
\delta_D=\sigma(a_D(\mathbf{y}))-1.
$$

把这一个梯度传入判别器的 `network_backward`，可得到判别器输入坐标的梯度 $\partial\mathcal{L}_G/\partial\mathbf{y}$。随后把该向量作为生成器反向传播的 `upstream`，继续得到 $\partial\mathcal{L}_G/\partial\theta$ 和 $\partial\mathcal{L}_G/\partial\mathbf{z}$。按链式法则：

$$
\frac{\partial\mathcal{L}_G}{\partial\mathbf{z}}
=\left(\frac{\partial G_{\theta}}{\partial\mathbf{z}}\right)^{\mathsf T}
\left(\frac{\partial D_{\psi}^{\mathrm{logit}}}{\partial\mathbf{y}}\right)^{\mathsf T}
\left(\sigma(a_D(\mathbf{y}))-1\right).
$$

$\psi$ 表示判别器参数，$D_{\psi}^{\mathrm{logit}}$ 表示 sigmoid 之前的判别器 logit。反传只读取 $\psi$ 来计算坐标导数，不意味着更新 $\psi$。`generator_backward` 把判别器参数梯度写入名为 `discarded` 的临时数组并丢弃，保留 `point_gradient` 作为生成器输出梯度；之后才调用生成器的 `network_backward`。若跳过判别器对输入的梯度，生成器就无法知道应往哪个方向移动点。

## 七、固定样本的完整数值链

接下来逐项重放 `test_gradients` 的固定夹具：生成器种子 $42$、判别器种子 $43$，测试向量 $(0.7,-0.4)$。判别器测试把它当作真实点；生成器测试则把同一数对当作固定潜在噪声。这里重复利用一个向量是为了可复现地检查导数，不能误解成训练时真实点与噪声配对。

### 判别器对固定真实点

网络前向给出 logit $a=0.056733650031$，于是

$$
p=\sigma(a)=0.514179609378.
$$

真实标签 $y=1$ 的 BCE 和 logit 梯度分别为

$$
\ell(a,1)=-\ln(0.514179609378)=0.665182639978,
\qquad
\frac{\partial\ell}{\partial a}=0.514179609378-1=-0.485820390622.
$$

把标签改为假样本 $y=0$，同一 logit 的损失为 $-\ln(1-p)=0.721916290009$，导数为 $p-0=0.514179609378$。对应的输入坐标导数在真实标签测试中是 $(-0.054044897509,-0.053368570237)$；输出偏置梯度等于 logit 梯度。数值都来自 `network_forward`、`binary_loss` 和 `network_backward` 的实际调用。

### 从生成坐标把梯度传回潜在噪声

上一节手算得到生成器输出
$\mathbf{y}=(-0.154483973954,0.107332626517)$。将其输入初始判别器后，logit 为 $a_D(\mathbf{y})=-0.014110946584$，判真概率为 $0.496472321889$。生成器用目标 $1$，所以

$$
\mathcal{L}_G=-\ln(0.496472321889)=0.700227543497,
\qquad
\delta_D=0.496472321889-1=-0.503527678111.
$$

判别器将这一个标量梯度反传到二维坐标，得到

$$
\frac{\partial\mathcal{L}_G}{\partial\mathbf{y}}
=(-0.067063434728,-0.032453019050).
$$

生成器输出层的偏置梯度正好等于这两个分量。对于第 $0$ 个输出、第 $0$ 个隐藏输入，权重梯度是输出误差乘以第二层第 $0$ 个激活：

$$
\frac{\partial\mathcal{L}_G}{\partial W^{(3)}_{0,0}}
=(-0.067063434728)(-0.025785502000)
=0.001729264330.
$$

两个因子都为负，所以乘积为正。这对应 `gradient[W3 + i * WIDTH + j] += upstream[i] * forward->second[j]`。再继续穿过生成器两层 `tanh`，得到潜在噪声梯度

$$
\frac{\partial\mathcal{L}_G}{\partial\mathbf{z}}
=(0.001717780203,-0.012000672188).
$$

这条链从判别器的“判真”方向开始，经判别器输入导数抵达生成坐标，再经生成器反向抵达潜在输入和生成器参数。**参数优化实际使用的是对生成器参数的梯度；噪声梯度是为了说明链式法则和验证反向实现而保留的诊断量。**

## 八、一次训练迭代的顺序

`train` 为生成器和判别器分别创建 Adam 状态、梯度数组和随机状态。公式中的批次索引 $b$ 从 $1$ 到 $B$，对应 C 循环变量 `sample` 的 $0$ 到 `BATCH_SIZE - 1`，关系为 $b=sample+1$。每一步先执行判别器阶段：对每个 $b\in\{1,\ldots,B\}$ 抽取一个真实点 $\mathbf{x}_b$ 和一个噪声 $\mathbf{z}_b$，用当前未更新的生成器产生假点，然后对真实点使用目标 $1$、假点使用目标 $0$：

$$
\nabla_{\psi}\mathcal{L}_D
=\frac{1}{B}\sum_{b=1}^{B}
\left[
\nabla_{\psi}\ell(a_D(\mathbf{x}_b),1)
+\nabla_{\psi}\ell(a_D(G_{\theta}(\mathbf{z}_b)),0)
\right].
$$

源码把每个样本的 logit 梯度先除以 `BATCH_SIZE`，再将梯度累加进 `dg`；循环结束后只调用一次 `adam_update` 更新 $D$。例如固定夹具中真实标签的单样本导数为 $-0.485820390622$，若只把它作为批量中的一项，除以 $B=32$ 后对该输出的贡献是 $-0.015181887207$；它不是完整批量梯度。生成器在此阶段保持不变，并通过 `memcmp` 断言验证。

随后代码复制更新后的判别器作为冻结快照，重新抽取一组生成器噪声 $\tilde{\mathbf{z}}_b$，计算

$$
\nabla_{\theta}\mathcal{L}_G
=\frac{1}{B}\sum_{b=1}^{B}
\nabla_{\theta}\ell(a_D(G_{\theta}(\tilde{\mathbf{z}}_b)),1),
\qquad
\psi\text{ 在此阶段固定}.
$$

每个样本的生成器梯度先放在局部 `gradient` 中，再除以 $B$ 加入批量数组 `gg`。全部样本完成后只更新一次生成器；判别器参数在这一阶段必须保持不变，代码再次用 `memcmp` 检查。真实点和两组潜在噪声使用各自独立的随机状态。

## 九、梯度裁剪与 Adam

`adam_update` 先计算整个参数梯度向量的欧氏范数 $\lVert\mathbf{g}\rVert_2$。若范数大于 $5$，就统一缩放所有分量：

$$
\mathbf{g}'=c\mathbf{g},
\qquad
c=\begin{cases}
5/\lVert\mathbf{g}\rVert_2,&\lVert\mathbf{g}\rVert_2>5,\\
1,&\lVert\mathbf{g}\rVert_2\le5.
\end{cases}
$$

零梯度时采用第二个分支，不会除以零。Adam 对每个参数保存一阶矩 $m_t$、二阶矩 $v_t$，并进行偏差修正：

$$
\begin{aligned}
m_t&=\beta_1m_{t-1}+(1-\beta_1)g'_t,\\
v_t&=\beta_2v_{t-1}+(1-\beta_2)(g'_t)^2,\\
\widehat{m}_t&=\frac{m_t}{1-\beta_1^t},
\qquad
\widehat{v}_t=\frac{v_t}{1-\beta_2^t},\\
w_t&=w_{t-1}-\eta\frac{\widehat{m}_t}{\sqrt{\widehat{v}_t}+\epsilon_{\mathrm{Adam}}}.
\end{aligned}
$$

这里 $t$ 是对应网络从 $1$ 开始的更新次数，$w_t$ 是某个参数，$g'_t$ 是裁剪后的该参数梯度；本例 $\eta=0.001$、$\beta_1=0.5$、$\beta_2=0.999$、$\epsilon_{\mathrm{Adam}}=10^{-8}$。`Adam.first`、`Adam.second` 保存 $m_t$、$v_t$，`beta1_power`、`beta2_power` 逐次保存衰减幂。生成器与判别器使用完全独立的优化器状态。

用第七节的单噪声夹具进一步手算第一次 Adam 更新。该完整生成器梯度范数是 $0.202493704228<5$，所以 $c=1$；输出偏置梯度 $g=-0.067063434728$。初始矩为零，故 $m_1=0.5g=-0.033531717364$、$v_1=0.001g^2\approx0.000004497504$，偏差修正后 $\widehat m_1=g$、$\widehat v_1=g^2$。于是偏置从 $0$ 更新为约 $0.000999999851$。这是固定单样本的一步推导；正式训练使用 $32$ 个样本的平均梯度。

更新不是逐参数直接写回：程序先复制 `Network candidate` 和 `Adam next`，检查梯度、范数、参数及矩估计全部有限后才提交。若发现 `NaN` 或无穷，函数返回失败，原参数和优化器状态都不变。这种先验证再提交的做法避免半个网络已更新、半个网络仍是旧值。

## 十、可重复训练、评估与模式崩塌

随机状态分工如下；每个状态各自推进，不会相互消耗：

| 随机用途 | 初始种子 | 代码位置 |
| --- | ---: | --- |
| 生成器初始化 | $42$ | `main` |
| 判别器初始化 | $43$ | `main` |
| 训练真实点 | $2025$ | `train` |
| 判别器阶段噪声 | $3025$ | `train` |
| 生成器阶段噪声 | $4025$ | `train` |
| 每千步监测 | $5025$ | `train` 调用 `evaluate` |
| 训练前后最终对照 | $6025$ | `main` |
| 散点图真实点 / 噪声 | $7025$ / $8025$ | `draw_scatter` |

`evaluate` 用 $1024$ 个生成点比较最近模式中心的欧氏距离。设第 $i$ 个生成点为 $\mathbf{y}_i$，最近中心索引为 $k_i$，则平均距离为

$$
\bar{d}=\frac{1}{1024}\sum_{i=1}^{1024}\min_{0\le k<8}\lVert\mathbf{y}_i-\mathbf{c}_k\rVert_2.
$$

距离不超过 $0.3$ 的点算作有效点；有效比例记为 $r_{\mathrm{valid}}$，其最近中心对应的簇计数记为 $n_k$。覆盖数的规则是

$$
r_{\mathrm{valid}}=\frac{\#\{i:\min_k\lVert\mathbf{y}_i-\mathbf{c}_k\rVert_2\le0.3\}}{1024},
\qquad
C=\sum_{k=0}^{7}\mathbf{1}[n_k\ge8].
$$

$\mathbf{1}[\cdot]$ 在条件为真时取 $1$，否则取 $0$。簇间过渡点不会增加任何簇计数。评估还报告真实点和生成点上的 $D$ 平均判真概率，但该数值不是概率密度。

这些指标依赖本案例已知的合成中心，只适合说明当前小实验。平均距离小但只覆盖一个簇仍可能是模式崩塌；覆盖 $8$ 个簇不代表每簇比例、方差和尾部都匹配。更换数据集后不能沿用“到这八个中心的距离”作为通用 GAN 指标，也不应只凭 $D$ 输出接近 $0.5$ 判断生成质量。

## 十一、测试能证明什么

`test_gradients` 使用中心有限差分检查解析梯度。对参数或输入中的某个标量 $v_i$，数值近似为

$$
g_i^{\mathrm{numeric}}
=\frac{\mathcal{L}(v_i+10^{-5})-\mathcal{L}(v_i-10^{-5})}{2\cdot10^{-5}}.
$$

判定条件为

$$
\left|g_i^{\mathrm{numeric}}-g_i^{\mathrm{analytic}}\right|
<2\cdot10^{-6}+2\cdot10^{-4}\left|g_i^{\mathrm{numeric}}\right|.
$$

判别器分别用目标 $0$ 和 $1$ 检查全部参数槽与两个输入坐标；生成器检查全部参数和两个固定噪声分量。每次正负扰动都固定其他输入和网络。梯度检查验证代码导数与当前离散实现一致，不证明训练一定稳定或对新分布泛化。

`test_boundaries` 还验证了：logit 为 $1000$、目标为 $0$ 时损失有限且等于 $1000$；logit 为 $-1000$、目标为 $1$ 时也得到 $1000$；相同种子会产生相同真实点；固定判别器和输入时，生成器一次 Adam 更新降低该样本损失；梯度含 `NaN` 时网络与 Adam 状态保持不变。训练结束后还断言独立评估的平均最近簇距离低于训练前。

本次运行 `make test-gan` 和 `make sanitize-gan` 均通过，输出为：

```text
独立评估：距离 1.5068→0.2342，有效=75.1%，覆盖=8/8，D(real)=0.499 D(fake)=0.500
有效簇计数：106 111 72 100 102 85 91 102（每簇至少8点才记作覆盖）
GAN数值验证通过：完整D→G链、输入/噪声导数、冻结D与更新边界。
```

这是固定种子、当前网络宽度和 $6000$ 步训练预算下的合成数据结果。它不保证换种子、换超参数或真实图像时仍得到同样质量。测试集没有被用于选模型；此实现没有模型保存、卷积网络、条件标签或高维图像数据。

## 十二、导出图片应该怎么看

`make gan` 导出四张 $320\times240$ BMP，坐标视图为 $[-3,3]\times[-3,3]$，图上 $y$ 轴向上。超出视图或非有限的点不绘制，但评估仍会将离群点计入到最近中心的距离。

- **`gan-before`**：绿色是固定真实点，橙色是同一组固定噪声经过初始生成器的输出。
- **`gan-after`**：真实点和噪声与训练前相同，橙色点替换成训练后生成器输出。
- **`gan-discriminator`**：颜色编码 $D$ 的判真概率；红色趋近 $1$，蓝色趋近 $0$。它显示判别器的分类场，不是数据密度图；没有样本支持的区域不能作真实分布解释。
- **`gan-latent-grid`**：把 $[-2,2]\times[-2,2]$ 上的 $25\times25$ 规则潜在网格送入 $G$，用输入列和行编码输出点颜色。它用于观察潜在空间到二维输出的映射，不是按正态分布抽样的密度图。

## 十三、可以继续做的实验

先保持判别器参数不变，固定同一噪声，观察一次生成器更新前后的 $G(\mathbf{z})$；再对照第七节中的坐标梯度，解释点为什么移动。随后可比较原始生成器目标与非饱和目标在很负 logit 时的导数，或一次只改变判别器/生成器的学习率、每步更新次数和隐藏层宽度之一。

每个实验都用多个初始化种子，比较有效比例、逐簇计数、最近中心距离与散点图。不要用最终评估反复选择超参数后仍把同一评估结果称作独立测试。这个二维案例建立了对抗训练和模式崩塌的基本直觉；图像 GAN 还需要卷积结构、图像数据处理和更严格的评估。
