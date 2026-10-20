# 纯 C VAE：二维潜在空间与图形生成

## 需求与验收

在已有16×16图形与自编码器基础上，实现可学习二维潜在分布的VAE。模型计算不依赖外部框架，沿用BMP导出；不引入窗口、模型文件、通用自动求导或分类条件。

| 编号 | 可观察结果 | 验证方式 | 状态 |
| --- | --- | --- | --- |
| AC-001 | 均值/方差、重参数化、Bernoulli重建与 KL 的全部导数一致 | 固定 $\boldsymbol{\epsilon}$ 中心差分、解析边界 | 通过 |
| AC-002 | 独立图形上的重建优于训练集平均图，训练/验证/测试各自生成 | 固定评估噪声及均值重建MSE | 通过 |
| AC-003 | 导出重建对照、二维网格、潜在插值及先验随机样本 | BMP导出与图像检查，不覆盖已有文件 | 通过 |
| AC-004 | 接入构建、短测试、消毒器、调试与文档；日志仅在logs内追加 | 运行检查及双输出验证 | 通过 |

实现使用 `src/ml/13_vae_shapes.c` 的固定大小数组，复用软件画布与BMP路径管理，不修改原自编码器行为。默认不产生文件；显式 `--export` 导出图片，`--log` 追加日志。

## 运行与观察

```sh
make vae                                  # 验证、训练并导出四张BMP
make test-vae                             # 相同验证/训练，不写文件
make sanitize-vae                         # 地址与未定义行为消毒器
./build/ml/13_vae_shapes                   # 只显示训练和测试指标
./build/ml/13_vae_shapes --export --log     # 图片与双输出日志
less logs/ml/13_vae_shapes.log
```

普通构建、`make ml`、`make test`、`make lint` 已包含本例。图片使用 `exports/` 中独占创建的时间/序号文件名，不覆盖旧文件；日志固定追加至 `logs/ml/13_vae_shapes.log`，创建父目录并带时间/源码位置前缀。`--test` 不能和文件输出选项合用；重复或未知选项报错。默认不写文件。

VS Code 选择 `C: ml/13_vae_shapes.c` 按 F5，默认带 `--export`。在 `main` 的 `train` 调用处断住，跳过数值差分循环；前向观察 `mu`、`logvar`、`sigma`、`epsilon`、`z`，反向观察 `dmu`、`dlv`、`draw`。普通构建使用 `-O2`，调试使用 `-O0 -g3`。模型计算仅用 ISO C 与数学库，图像目录及日志目录创建复用项目 POSIX 模块。

## 1. 从确定编码到概率编码

普通自编码器为每张图输出固定潜在坐标。VAE 则编码一个对角高斯分布：

$$
q_{\phi}(\mathbf{z}\mid\mathbf{x}) =
\mathcal{N}\!\left(\boldsymbol{\mu}(\mathbf{x}),
\operatorname{diag}\!\left(\exp(\boldsymbol{\ell}(\mathbf{x}))\right)\right),
\qquad \ell_k = \log \sigma_k^2.
$$

两个输出分支都可学习；$\ell_k$ 表示方差的对数，不是标准差的对数 $\log\sigma_k$。

$$
\begin{aligned}
\mathbf{x}\in[0,1]^{256}
&\longrightarrow \mathbf{h}^{(e)}\in\mathbb{R}^{32}
\longrightarrow (\boldsymbol{\mu},\boldsymbol{\ell})\in\mathbb{R}^2\times\mathbb{R}^2,\\
\mathbf{z}&=\boldsymbol{\mu}+\exp(\tfrac12\boldsymbol{\ell})\odot\boldsymbol{\epsilon},\\
\mathbf{z}\in\mathbb{R}^2
&\longrightarrow \mathbf{h}^{(d)}\in\mathbb{R}^{32}
\longrightarrow \hat{\mathbf{x}}\in[0,1]^{256}.
\end{aligned}
$$

其中两个隐藏向量都使用 $\tanh$，输出向量使用 sigmoid。

共16900个参数，权重按 `[输出][输入]` 行优先排列。固定容量数组由调用方拥有，无动态分配。独立训练/验证/测试图形为96/48/48张，均衡包含圆、方、三角，位置和尺寸采用阶段4的同类变化范围；本例没有加入去噪任务。

训练图形种子 2025，验证 2026，测试 2027，参数初始化 42。训练噪声随机状态从 4025 持续推进；Box–Muller 使用开区间 $(0,1)$ 均匀数，避免 $\log(0)$，每次产生两个标准正态值。

伪随机状态使用 32 位线性同余递推，随后取高 24 位构造开区间均匀数：

$$
s_{n+1}=(1664525s_n+1013904223)\bmod 2^{32},
\qquad
U=\frac{(s_{n+1}\mathbin{\texttt{>>}}8)+0.5}{2^{24}}.
$$

其中 $s_n$ 是第 $n$ 次状态，$U$ 是均匀随机数。给定两个独立的 $U_1,U_2$，Box–Muller 变换为

$$
\rho=\sqrt{-2\log U_1},\qquad \vartheta=2\pi U_2,
\qquad \epsilon_0=\rho\cos\vartheta,\quad \epsilon_1=\rho\sin\vartheta.
$$

其中 $\rho$ 是标准正态样本的半径，$\vartheta$ 是角度。
例如边界测试以状态 $s_0=7$ 开始，得到 $U_1\approx0.238780826330$、$U_2\approx0.913493245840$，进而得到 $(\epsilon_0,\epsilon_1)\approx(1.448549354342,-0.875284612517)$。随机数只用于确定性教学实验，不适合安全场景。

## 2. 为什么需要重参数化

直接把随机抽样作为一个黑盒，无法沿抽样操作直接回传参数导数。改写为

$$
\mathbf{z}=\boldsymbol{\mu}+\boldsymbol{\sigma}\odot\boldsymbol{\epsilon},
\qquad \boldsymbol{\epsilon}\sim\mathcal{N}(\mathbf{0},I)
$$

后，将随机性放在独立的 $\boldsymbol{\epsilon}$ 中，网络仍通过确定的运算接收导数。

一次前向和反向使用同一 $\boldsymbol{\epsilon}$；数值差分的正负扰动也必须使用相同噪声，否则损失差含随机噪声，无法验证导数。训练时每个样本重新采样；推理可解码均值 $\boldsymbol{\mu}$（令 $\boldsymbol{\epsilon}=\mathbf{0}$），或继续从后验采样。**解码 $\boldsymbol{\mu}$ 不等于解码输出的期望**，因为解码器是非线性的。

为限制指数运算范围，输出分支使用光滑参数化

$$
\ell=6\tanh(r/6),\qquad |\ell|<6\quad\text{(实数运算)}.
$$

其中 $r$ 是方差分支的原始线性输出。它限制了可表示的后验方差，也可能在饱和处使导数变小；不是没有影响的兼容技巧。浮点计算可能把极端输入舍入到 $\ell=\pm6$，所以代码边界测试采用闭区间 $[-6,6]$。反向必须乘
$\mathrm{d}\ell/\mathrm{d}r=1-(\ell/6)^2$，不能忽略这一层或当作硬裁切。

## 3. 重建似然、KL与负ELBO

输出 sigmoid 表示每个像素为 1 的概率，不直接随机采样黑白像素；展示灰度概率便于观察不确定区域。假设给定 $\mathbf{z}$ 后像素条件独立，重建项为每图像素 BCE 之和。令 $u_j$ 是像素 $j$ 的 logit，$\hat{x}_j=\operatorname{sigmoid}(u_j)$：

$$
\begin{aligned}
\mathcal{R}(\mathbf{x},\mathbf{u})
&=\sum_{j=0}^{255}
  \left[\operatorname{softplus}(u_j)-x_j u_j\right],\\
\operatorname{softplus}(u)
&=\max(u,0)+\log\!\left(1+e^{-|u|}\right),\\
\mathcal{K}(\mathbf{x})
&=\frac12\sum_{k=0}^{1}
  \left[\mu_k^2+e^{\ell_k}-1-\ell_k\right],\\
\mathcal{J}
&=\mathcal{R}+\mathcal{K}.
\end{aligned}
$$

这里 $\mathcal{R}$ 是重建负对数似然，$\mathcal{K}$ 是 KL 惩罚，$\mathcal{J}$ 是待最小化的负 ELBO。

这是标准正态先验下、KL 权重 $\beta_{\mathrm{KL}}=1$ 的负 ELBO 单次采样估计。重建项鼓励解释输入，KL 惩罚后验偏离标准正态，便于从先验生成。KL 不是让所有样本“强制完全相同”的规则；两项需要平衡。若把 $\mathcal{R}$ 改成平均像素 BCE，却仍加同一个 $\mathcal{K}$，相当于大幅加强 KL 的相对权重，不是单纯换显示单位。

代码用稳定 softplus 计算 BCE，避免对下溢概率取对数；KL 使用 `expm1(logvar) - logvar`，即数学式 $\operatorname{expm1}(\ell_k)-\ell_k$，改善 $\ell_k$ 靠近零时的消减误差。MSE 只用于直观比较重建，并不参与训练。

VAE的概率建模和重参数化原理可参考[原始论文](https://arxiv.org/abs/1312.6114)。本例是固定合成图形上的小型教学实现。

## 4. 反向传播与验证

设解码器传回潜变量梯度 $g_{z_k}=\partial\mathcal{R}/\partial z_k$。重建项和 KL 项分别贡献以下导数：

$$
\begin{aligned}
\frac{\partial\mathcal{J}}{\partial\mu_k}
&=g_{z_k}+\mu_k,\\
\frac{\partial\mathcal{J}}{\partial\ell_k}
&=g_{z_k}\,\frac12\sigma_k\epsilon_k
  +\frac12\left(e^{\ell_k}-1\right),\\
\frac{\partial\mathcal{J}}{\partial r_k}
&=\frac{\partial\mathcal{J}}{\partial\ell_k}
  \left[1-(\ell_k/6)^2\right],\\
\frac{\partial\mathcal{J}}{\partial\epsilon_k}
&=g_{z_k}\sigma_k.
\end{aligned}
$$

其中 $r_k$ 是方差头经线性层得到、再传入有界参数化的原始值；$u_j$ 则是解码器输出像素的 logit。

前一部分来自重参数化重建路径，后一部分来自KL；均值和方差分支同时回到编码器隐藏层。完成整个批次反向后才更新参数。

内置中心差分检查全部 16900 个参数、256 个输入以及 2 个 $\boldsymbol{\epsilon}$ 分量的导数。输入检查对同一个 $\mathbf{x}$ 同时作为编码输入和重建目标的总函数求导，因此包含 BCE 目标项的 $-u_j$ 导数；这不同于“固定目标，仅扰动编码器输入”。检查使用区间内部灰度值，避免扰动越过 $[0,1]$。

解析断言覆盖标准正态后验时 $\mathrm{KL}=0$、$\mu_k\neq0$ 时的 KL、重参数化、$\ell_k$ 上下边界、稳定像素 logit、采样可重复、裁剪和非有限更新失败时不部分修改。

## 5. 训练与独立评估

使用 250 轮、批量 8、Adam 学习率 0.003、全局梯度裁剪阈值 5。每个训练样本每次使用新的 $\boldsymbol{\epsilon}$。固定验证随机种子 5025，每张图采样 4 次，按平均 $\mathcal{R}+\mathcal{K}$ 选择检查点；KL 可解析计算，不靠抽样估计。最终测试同样采用可重复评估噪声，不参与参数或检查点选择。

另外令 $\boldsymbol{\epsilon}=\mathbf{0}$ 解码均值，计算确定性重建 MSE，并与仅从训练集计算的平均图比较。该 MSE 不是后验随机重建的期望误差。合成图形各组独立生成，但形状类别和变化范围共享，不能据此证明真实图像的生成能力。

MSE 的含义可以用两个像素快速核算：若目标是 $(1,0)$，预测是 $(0.8,0.2)$，则

$$
\operatorname{MSE}=\frac{(1-0.8)^2+(0-0.2)^2}{2}=0.04.
$$

本例把分母换成像素数 256，并对整张均值重建计算；这个指标与 BCE 重建 NLL 的单位和用途不同。

## 6. 四张图片怎样阅读

所有图均为320×240，不单独拉伸每个小图的对比度。

- **vae-reconstruction**：三行是圆/方/三角；左列原图、中列解码均值 $\boldsymbol{\mu}$、右列后验随机采样重建，展示采样变化。
- **vae-latent-grid**：7×7 小图，横轴 $z_0\in[-2,2]$，纵轴 $z_1$ 从上方 2 到下方 $-2$。每格直接解码该坐标，跳过编码器；这是规则网格，不是随机样本或分类标签地图。
- **vae-interpolation**：8 幅图从左到右、再换行，线性插值第一张圆与第三张三角的编码均值 $\boldsymbol{\mu}$。只插值潜在坐标，不对输出图片做像素混合。
- **vae-prior-samples**：12个标准正态先验坐标直接解码，未使用测试图进行编码。导出随机状态从6025开始，与训练及评估分离。

二维瓶颈和简单Bernoulli解码器可能生成模糊边界或混合图形；不能保证每个网格坐标都对应清晰的某一类别。这里展示的是潜在空间行为，生成图的视觉效果不能替代独立评估。

## 7. 实测结果与验证

在 macOS ARM64、Apple Clang 17、C23、默认种子和参数下，验证集选择第243轮检查点：

| 指标 | 结果 |
| --- | ---: |
| 验证集负ELBO | 17.5418 |
| 独立测试重建NLL | 14.5309 |
| 独立测试KL | 3.3248 |
| 独立测试负ELBO | 17.8557 |
| 独立测试均值重建MSE | 0.01348 |
| 训练集平均图在测试集上的MSE | 0.03990 |

`make test-vae`、`make sanitize-vae`、仓库整体 `make test` 与 `make lint` 均通过。四张BMP已检查尺寸、排列、灰度显示与生成内容；潜在网格可观察到圆、方、三角及过渡形状。`--log` 已验证保留原文件内容、追加输出与屏幕逐字一致并含源码位置；非法与重复选项被拒绝。调试程序构建和 `make compile-commands` 通过，编译数据库位于 `build/compile_commands.json`。

这些指标只验证当前合成数据、固定种子和小型网络的学习行为。更换数据、种子或编译器时，具体数值可能变化。

## 8. 练习

先固定 $\boldsymbol{\epsilon}$ 手算 $\boldsymbol{\mu}$ 与 $\boldsymbol{\sigma}$ 的导数；再比较同一图的多次后验重建。尝试改变 KL 权重、$\ell_k$ 范围或潜在维度，同时修改采样与数组形状并重新做梯度检查。用验证集选择配置，保留最终测试集；关注重建质量、KL 及先验样本的共同变化，不只追求训练 MSE。

## 9. 源码结构与数据流

模型集中在一个 C 文件中，固定容量数组使维度和内存布局容易检查。训练无需动态分配、外部机器学习框架或自动求导；图形导出才依赖项目的画布与 BMP 模块。

| 阶段 | 源码位置 | 阅读重点 |
| --- | --- | --- |
| 维度与参数偏移 | [`13_vae_shapes.c:13-74`](../src/ml/13_vae_shapes.c#L13-L74) | 16×16 输入、32 单元隐藏层、2 维潜变量及前向缓存 |
| 随机数、数据与初始化 | [`13_vae_shapes.c:105-171`](../src/ml/13_vae_shapes.c#L105-L171) | 可复现 LCG、Box–Muller、三类二值图与 Xavier 权重 |
| 解码与前向传播 | [`13_vae_shapes.c:194-259`](../src/ml/13_vae_shapes.c#L194-L259) | Bernoulli logits、后验参数、重参数化和 KL |
| 反向传播 | [`13_vae_shapes.c:270-320`](../src/ml/13_vae_shapes.c#L270-L320) | 输出误差如何传回解码器、潜变量和编码器 |
| 梯度裁剪与 Adam | [`13_vae_shapes.c:329-382`](../src/ml/13_vae_shapes.c#L329-L382) | 全局范数、偏差修正、失败时不部分提交 |
| 评估与训练 | [`13_vae_shapes.c:406-488`](../src/ml/13_vae_shapes.c#L406-L488) | 固定验证噪声、均值重建、批量更新和最佳检查点 |
| 数值验证 | [`13_vae_shapes.c:493-579`](../src/ml/13_vae_shapes.c#L493-L579) | 中心差分、解析边界和失败原子性 |
| 输出与入口 | [`13_vae_shapes.c:625-774`](../src/ml/13_vae_shapes.c#L625-L774) | 潜变量图像、选项解析、数据划分及最终断言 |

`dataset_generate` 每三张图循环生成圆、方、三角，网络本身不接收类别标签。中心坐标和半径由三个独立的均匀数控制：

$$
c_x=6.75+1.5U_1,\qquad
c_y=6.75+1.5U_2,\qquad
R=3+U_3,
\qquad U_1,U_2,U_3\in(0,1).
$$

像素坐标 $(c,v)\in\{0,\ldots,15\}^2$ 对应 C 变量 `column` 和 `row`；偏移为 $d_x=c-c_x$、$d_y=v-c_y$。圆形条件是 $d_x^2+d_y^2\leq R^2$，方形条件是 $|d_x|\leq R$ 且 $|d_y|\leq R$，三角形条件是 $-R\leq d_y\leq R$ 且 $|d_x|\leq(d_y+R)/2$。例如令三个随机数都为 $0.5$，则中心为 $(7.5,7.5)$、半径为 $3.5$；像素 $(7,7)$ 的偏移为 $(-0.5,-0.5)$，满足圆内条件 $0.5\leq12.25$。因为样本类别按索引循环、每组用独立固定种子生成，训练、验证、测试图像不同，但共享相同的三类分布。

权重使用 Xavier 均匀初始化。若一层有 $n_{\mathrm{in}}$ 个输入和 $n_{\mathrm{out}}$ 个输出，代码令

$$
s=\sqrt{\frac{6}{n_{\mathrm{in}}+n_{\mathrm{out}}}},
\qquad W_{ij}\sim\operatorname{Uniform}(-s,s),
\qquad b_i=0.
$$

例如第一层 $n_{\mathrm{in}}=256$、$n_{\mathrm{out}}=32$，所以 $s=\sqrt{6/288}=0.144337567$。此范围控制不同层激活和梯度的初始尺度；随机种子 42 使权重序列可重复。

`Network.parameters` 是一个长度为 16900 的连续 `double` 数组。下表中的起始位置以零为基准，区间采用左闭右开形式；权重按 `[输出单元][输入单元]` 连续存放，偏置紧随权重。

| C 偏移 | 起始位置 | 数量 | 对应映射 |
| --- | ---: | ---: | --- |
| `EW` / `EB` | 0 / 8192 | 8192 / 32 | 输入 256 到编码器 32 |
| `MW` / `MB` | 8224 / 8288 | 64 / 2 | 编码器 32 到均值 2 |
| `LW` / `LB` | 8290 / 8354 | 64 / 2 | 编码器 32 到原始对数方差 2 |
| `DW` / `DB` | 8356 / 8420 | 64 / 32 | 潜变量 2 到解码器 32 |
| `OW` / `OB` | 8452 / 16644 | 8192 / 256 | 解码器 32 到像素 256 |

参数总数可以直接从各矩阵和偏置相加核对：

$$
\begin{aligned}
P={}&(256\cdot32+32)
+2(32\cdot2+2)
+(2\cdot32+32)
+(32\cdot256+256)\\
={}&16900.
\end{aligned}
$$

## 10. 从概率模型推导训练目标

令 $\mathbf{x}\in\{0,1\}^{256}$ 为输入图像，$\mathbf{z}\in\mathbb{R}^2$ 为潜变量。编码器参数记为 $\phi$，解码器参数记为 $\theta$；源码把两者打包在同一个 `Network.parameters` 数组里。先验是标准正态 $p(\mathbf{z})=\mathcal{N}(\mathbf{0},I)$，编码器近似后验为

$$
q_{\phi}(\mathbf{z}\mid\mathbf{x})
=\mathcal{N}\!\left(\boldsymbol{\mu},
\operatorname{diag}(\boldsymbol{\sigma}^2)\right).
$$

图像的边缘概率要对所有潜变量积分，直接计算通常不可行。引入近似后验并使用 Jensen 不等式：

$$
\begin{aligned}
\log p_{\theta}(\mathbf{x})
&=\log\mathbb{E}_{q_{\phi}(\mathbf{z}\mid\mathbf{x})}
  \left[\frac{p_{\theta}(\mathbf{x}\mid\mathbf{z})p(\mathbf{z})}
  {q_{\phi}(\mathbf{z}\mid\mathbf{x})}\right]\\
&\geq \mathbb{E}_{q_{\phi}}
  [\log p_{\theta}(\mathbf{x}\mid\mathbf{z})]
  -D_{\mathrm{KL}}\!\left(q_{\phi}(\mathbf{z}\mid\mathbf{x})\,\|\,p(\mathbf{z})\right)
  =\mathcal{E}(\mathbf{x}).
\end{aligned}
$$

$\mathcal{E}$ 是证据下界 ELBO。最大化它等价于最小化负 ELBO：

$$
\mathcal{J}(\mathbf{x})=-\mathcal{E}(\mathbf{x})
=\mathbb{E}_{q_{\phi}}[-\log p_{\theta}(\mathbf{x}\mid\mathbf{z})]
+\beta_{\mathrm{KL}}D_{\mathrm{KL}}(q_{\phi}\|p),
\qquad \beta_{\mathrm{KL}}=1.
$$

给定 $\mathbf{z}$ 后，解码器假设每个像素条件独立，并输出像素为 1 的概率 $\hat{x}_j=\operatorname{sigmoid}(u_j)$：

$$
p_{\theta}(\mathbf{x}\mid\mathbf{z})
=\prod_{j=0}^{255}\hat{x}_j^{x_j}(1-\hat{x}_j)^{1-x_j}.
$$

因此负对数似然可逐像素相加：

$$
-\log p_{\theta}(\mathbf{x}\mid\mathbf{z})
=\sum_{j=0}^{255}\left[-x_j\log\hat{x}_j-(1-x_j)\log(1-\hat{x}_j)\right].
$$

sigmoid 与 Bernoulli 负对数似然可以先化简，再实现稳定计算。因为

$$
\log\operatorname{sigmoid}(u)=u-\operatorname{softplus}(u),
\qquad
\log(1-\operatorname{sigmoid}(u))=-\operatorname{softplus}(u),
$$

代入 Bernoulli 负对数似然后，得到

$$
-x\log\hat{x}-(1-x)\log(1-\hat{x})
=\operatorname{softplus}(u)-xu.
$$

代码直接从 logit $u_j$ 计算等价的稳定形式，而不先算概率再取对数：

$$
\operatorname{BCE}(x_j,u_j)
=\max(u_j,0)-x_j u_j+\log(1+e^{-|u_j|}).
$$

这对应 `decode` 中的 `fmax(value, 0) - target[i] * value + log1p(exp(-fabs(value)))`。`value` 是输出 logit，`target[i]` 是像素目标；此处 `log1p` 能更准确地计算接近零的 $\log(1+y)$。

对角高斯与标准正态之间的 KL 可以解析积分。对每个潜变量维度 $k$，令
$q_k(z_k)=\mathcal{N}(\mu_k,\sigma_k^2)$、$p_k(z_k)=\mathcal{N}(0,1)$：

$$
\begin{aligned}
D_{\mathrm{KL}}(q_k\|p_k)
&=\frac12\left(\mu_k^2+\sigma_k^2-1-\log\sigma_k^2\right)\\
&=\frac12\left(\mu_k^2+e^{\ell_k}-1-\ell_k\right),
\qquad \ell_k=\log\sigma_k^2.
\end{aligned}
$$

推导中用到正态分布的两个矩：$\mathbb{E}_{q_k}[(z_k-\mu_k)^2]=\sigma_k^2$，$\mathbb{E}_{q_k}[z_k^2]=\mu_k^2+\sigma_k^2$。将它们代入

$$
D_{\mathrm{KL}}(q_k\|p_k)
=\mathbb{E}_{q_k}[\log q_k(z_k)-\log p_k(z_k)]
$$

即可消去积分。以数值例中的第 0 维为例，代入 $\mu_0=0.404191760556$、$e^{\ell_0}=0.487129466705$ 和 $\ell_0=-0.719225345840$：

$$
\mathcal{K}_0
=\frac12\left(0.404191760556^2+0.487129466705-1+0.719225345840\right)
=0.184862895923.
$$

总 KL 对两个维度求和。源码使用 `expm1(ell) - ell`，因为 `expm1(ell)` 计算 $e^{\ell}-1$，在 $\ell$ 接近零时比先算指数再减 1 更能保留有效数字。训练时重建项用一次随机 $\mathbf{z}$ 估计期望，KL 则是解析值；所以优化目标是 Monte Carlo 估计的负 ELBO。

### 公式中的符号

| 符号 | 含义 | 对应代码或范围 |
| --- | --- | --- |
| $\mathbf{x}$、$x_j$ | 输入/目标图像及第 $j$ 个像素 | `input`；真实数据为 0 或 1，梯度夹具也测试 0.1、0.4、0.7 |
| $j$ | 像素索引 | $0\leq j<256$，对应 16×16 行优先像素 |
| $i$ | 隐藏单元或参数索引 | 隐藏层内 $0\leq i<32$，全局参数索引 $0\leq i<P$ |
| $\phi,\theta$ | 编码器参数、解码器参数 | 同属 `Network.parameters` |
| $I$ | 标准正态先验的单位矩阵 | 2×2 单位矩阵 |
| $\mathbf{h}^{(e)}$ | 编码器的 32 维 tanh 输出 | `Forward.encoder` |
| $\boldsymbol{\mu}$ | 后验均值向量 | `Forward.mu`，有 2 个分量 |
| $r_k$、$\ell_k$ | 原始对数方差头输出、经 tanh 限幅的对数方差 | `raw`、`logvar`；$\ell_k=6\tanh(r_k/6)$ |
| $\sigma_k$ | 后验标准差 | `Forward.sigma`；$\sigma_k=e^{\ell_k/2}>0$ |
| $\epsilon_k$ | 标准正态噪声 | `Forward.epsilon`；与参数无关，训练采样、差分时固定 |
| $z_k$ | 实际送给解码器的潜变量 | `Forward.z`；$z_k=\mu_k+\sigma_k\epsilon_k$ |
| $\mathbf{h}^{(d)}$ | 解码器的 32 维 tanh 输出 | `Forward.decoder` |
| $u_j$、$\hat{x}_j$ | 像素 logit、像素为 1 的预测概率 | `decode` 的 `value`、`Forward.image[j]` |
| $\mathcal{R}$、$\mathcal{K}$、$\mathcal{J}$ | 重建 NLL、KL、负 ELBO | `Forward.reconstruction`、`Forward.kl`、二者之和 |
| $g_{z_k}$ | 重建损失沿解码器回传到潜变量的导数 | 由 `network_backward` 累加的 `dz[k]` |
| $P$、$G$ | 参数总数、裁剪前全局梯度范数 | $P=16900$；$G=\sqrt{\sum_{i=0}^{P-1}g_i^2}$ |
| $m_t,v_t$ | Adam 一阶、二阶梯度矩估计 | `Adam.first`、`Adam.second` |
| $\beta_{\mathrm{KL}}$ | KL 项权重 | 本例固定为 1 |
| $\beta_1,\beta_2,\eta,t$ | Adam 衰减率、学习率和更新步 | $0.9$、$0.999$、$0.003$ 及当前批次序号 |
| $g'_t,w_t$ | 裁剪后的参数梯度分量、该参数更新后的值 | `adam_update` 的输入与候选参数 |

## 11. 用梯度检查夹具手算一次前向

[`test_gradients`](../src/ml/13_vae_shapes.c#L493-L535) 提供了可复现的具体输入：参数用种子 42 初始化，像素按 $x_j=0.1+0.6(j\bmod3)/2$ 生成，因此前三个像素依次是 $0.1,0.4,0.7$；固定噪声为 $\boldsymbol{\epsilon}=(0.7,-0.4)$。下面数值来自按同一组参数直接运行 `network_forward`，用于把公式和源码的真实中间量对应起来。这是梯度检查夹具，不是训练集中的二值图。

### 编码器与潜变量

编码器第 0 个隐藏单元先做 256 项点积，再过 tanh：

$$
a^{(e)}_0=b^{(e)}_0+\sum_{j=0}^{255}W^{(e)}_{0j}x_j
=0.513931232880,
\qquad h^{(e)}_0=\tanh(a^{(e)}_0)=0.473002560852.
$$

这里 $a^{(e)}_0$ 是编码器预激活值，$b^{(e)}_0$ 和 $W^{(e)}_{0j}$ 分别是该单元偏置和第 $j$ 个输入权重。源码的内层循环正是将每项 $W^{(e)}_{0j}x_j$ 累加到 `value`，见 [`network_forward` 第 233-239 行](../src/ml/13_vae_shapes.c#L233-L239)。

均值头与方差头共享 $\mathbf{h}^{(e)}$，但拥有不同权重。潜变量第 0 维得到：

$$
\begin{aligned}
\mu_0&=b^{(\mu)}_0+\sum_{i=0}^{31}W^{(\mu)}_{0i}h^{(e)}_i
=0.404191760556,\\
r_0&=b^{(\ell)}_0+\sum_{i=0}^{31}W^{(\ell)}_{0i}h^{(e)}_i
=-0.722700210642,\\
\ell_0&=6\tanh(r_0/6)=-0.719225345840,\\
\sigma_0&=e^{\ell_0/2}=e^{-0.359612672920}=0.697946607346.
\end{aligned}
$$

将固定噪声代入重参数化：

$$
z_0=\mu_0+\sigma_0\epsilon_0
=0.404191760556+0.697946607346\cdot0.7
=0.892754385698.
$$

第二维同理：$\mu_1=-0.264211385525$、$\ell_1=0.833225257646$、$\sigma_1=1.516814828771$、$\epsilon_1=-0.4$，于是 $z_1=-0.870937317034$。注意标准差可以大于 1；KL 项会惩罚它偏离先验的单位标准差。

对应代码是 [`network_forward` 第 240-257 行](../src/ml/13_vae_shapes.c#L240-L257)：先算两组头部，再限幅 $\ell_k$，接着求 $\sigma_k$、$z_k$，并逐维累加解析 KL。`Forward` 把这些中间值保留下来，反向传播无需重新计算或重新采样。

### 解码器、一个像素和总损失

解码器第 0 个隐藏单元的线性输入和激活为

$$
a^{(d)}_0=b^{(d)}_0+W^{(d)}_{00}z_0+W^{(d)}_{01}z_1
=0.004399731509,
\qquad h^{(d)}_0=\tanh(a^{(d)}_0)=0.004399703120.
$$

经过全部 32 个解码单元后，像素 0 的输出 logit 是 $u_0=0.078081186751$，所以

$$
\hat{x}_0=\operatorname{sigmoid}(u_0)=0.519510385327.
$$

目标像素是 $x_0=0.1$，这一个像素的 BCE 为

$$
\begin{aligned}
\operatorname{BCE}(x_0,u_0)
&=\max(u_0,0)-x_0u_0+\log(1+e^{-|u_0|})\\
&=0.078081186751-0.1\cdot0.078081186751
  +\log(1+e^{-0.078081186751})\\
&=0.725141545714.
\end{aligned}
$$

其余 255 个像素按同一公式累加，得到重建 NLL $\mathcal{R}=179.814382408349$。两个 KL 分量分别是 $\mathcal{K}_0=0.184862895923$ 和 $\mathcal{K}_1=0.268654811687$，因此

$$
\mathcal{K}=\mathcal{K}_0+\mathcal{K}_1=0.453517707611,
\qquad \mathcal{J}=\mathcal{R}+\mathcal{K}=180.267900115960.
$$

这是单张灰度夹具、单个固定噪声下的损失，不能和验证集每图平均 NLL 直接比较。训练图像是 0/1 值，而此夹具的灰度输入特意用来让有限差分扰动保持在合法区间。

## 12. 从一个像素误差推回网络参数

输出使用 sigmoid 概率和 BCE 的组合，其对 logit 的导数会简化为概率减目标：

$$
\delta^{(o)}_j
=\frac{\partial\mathcal{R}}{\partial u_j}
=\hat{x}_j-x_j.
$$

这项抵消可以逐步核对。sigmoid 导数为
$\mathrm{d}\hat{x}_j/\mathrm{d}u_j=\hat{x}_j(1-\hat{x}_j)$，而 BCE 对概率的导数为

$$
\frac{\partial\operatorname{BCE}}{\partial\hat{x}_j}
=-\frac{x_j}{\hat{x}_j}+\frac{1-x_j}{1-\hat{x}_j}.
$$

两者相乘后，$-x_j(1-\hat{x}_j)+(1-x_j)\hat{x}_j=\hat{x}_j-x_j$。这也是直接对 logit 算稳定 BCE 的优点：反向传播不需要显式除以接近 0 的概率。
对上面的像素 0，$\delta^{(o)}_0=0.519510385327-0.1=0.419510385327$。因此输出偏置梯度就是该误差；权重梯度再乘对应的解码器输入：

$$
\frac{\partial\mathcal{J}}{\partial OB_0}=0.419510385327,
\qquad
\frac{\partial\mathcal{J}}{\partial OW_{0,0}}
=\delta^{(o)}_0h^{(d)}_0=0.001845721151.
$$

这对应 [`network_backward` 第 276-294 行](../src/ml/13_vae_shapes.c#L276-L294) 中的输出残差、`gradient[OW + ...]` 和解码器梯度累加。`OW` 的第 0 行属于像素 0，第 0 列连到解码器第 0 个单元。

把 `OW`、`DW`、`MW`、`LW`、`EW` 分别记作输出层、解码层、均值头、原始对数方差头和编码层矩阵。偏置梯度等于该层的误差向量；权重梯度是误差与该层输入的外积。反向传播的完整向量关系是

$$
\begin{aligned}
\mathbf{g}_{h^{(d)}}&=(W^{(o)})^{\mathsf T}\boldsymbol{\delta}^{(o)},\\
\boldsymbol{\delta}^{(d)}
&=\mathbf{g}_{h^{(d)}}\odot\left(\mathbf{1}-(\mathbf{h}^{(d)})^2\right),\\
\nabla_{W^{(d)}}\mathcal{J}&=\boldsymbol{\delta}^{(d)}\mathbf{z}^{\mathsf T},\\
\mathbf{g}_{\mathbf{z}}&=(W^{(d)})^{\mathsf T}\boldsymbol{\delta}^{(d)},\\
\mathbf{g}_{\mathbf{h}^{(e)}}
&=(W^{(\mu)})^{\mathsf T}\mathbf{g}_{\boldsymbol{\mu}}
 +(W^{(r)})^{\mathsf T}\mathbf{g}_{\mathbf{r}},\\
\boldsymbol{\delta}^{(e)}
&=\mathbf{g}_{\mathbf{h}^{(e)}}\odot\left(\mathbf{1}-(\mathbf{h}^{(e)})^2\right),\\
\nabla_{W^{(e)}}\mathcal{J}&=\boldsymbol{\delta}^{(e)}\mathbf{x}^{\mathsf T}.
\end{aligned}
$$

这里 $\mathsf T$ 表示转置，$\odot$ 表示逐元素乘法，$\mathbf{1}$ 是全 1 向量；隐藏层 tanh 的导数为 $1-h_i^2$。均值头和方差头权重梯度分别为 $\mathbf{g}_{\boldsymbol{\mu}}(\mathbf{h}^{(e)})^{\mathsf T}$ 与 $\mathbf{g}_{\mathbf{r}}(\mathbf{h}^{(e)})^{\mathsf T}$。这些矩阵乘法在 C 中由嵌套循环展开，和 [`network_backward`](../src/ml/13_vae_shapes.c#L270-L320) 中的索引式一一对应。

沿解码器权重继续回传可得到 $g_{z_0}=1.346639685250$。它是重建 NLL 的导数，不包含 KL；随后均值和对数方差分支分别合并重建路径与 KL 路径：

$$
\begin{aligned}
\frac{\partial\mathcal{J}}{\partial\mu_0}
&=g_{z_0}+\mu_0\\
&=1.346639685250+0.404191760556\\
&=1.750831445806,\\[3pt]
\frac{\partial\mathcal{J}}{\partial\ell_0}
&=g_{z_0}\frac12\sigma_0\epsilon_0
  +\frac12(e^{\ell_0}-1)\\
&=0.328958909873-0.256435266647\\
&=0.072523643226,\\[3pt]
\frac{\partial\mathcal{J}}{\partial r_0}
&=\frac{\partial\mathcal{J}}{\partial\ell_0}
  \left(1-(\ell_0/6)^2\right)\\
&=0.072523643226\cdot0.985630969497\\
&=0.071481548784,\\[3pt]
\frac{\partial\mathcal{J}}{\partial\epsilon_0}
&=g_{z_0}\sigma_0=0.939882599637.
\end{aligned}
$$

第二行中的第一项来自 $z_0$ 对 $\ell_0$ 的影响，第二项来自解析 KL；这解释了为什么方差头的梯度可能出现两项相互抵消。代码将 $\partial\mathcal{J}/\partial\mu_0$ 写入 `gradient[MB]`，将 $\partial\mathcal{J}/\partial r_0$ 写入 `gradient[LB]`，中间的 $\partial\mathcal{J}/\partial\ell_0$ 则只存于局部变量 `dlv`。均值头首权重的梯度为

$$
\frac{\partial\mathcal{J}}{\partial MW_{0,0}}
=1.750831445806\cdot0.473002560852
=0.828147757486,
$$

方差头首权重的梯度为 $\partial\mathcal{J}/\partial LW_{0,0}=0.071481548784\cdot0.473002560852=0.033810955628$。这两项分别对应 [`network_backward` 第 297-310 行](../src/ml/13_vae_shapes.c#L297-L310)。

还有一个容易忽略的细节：梯度检查同时扰动输入和重建目标，因为同一个 `input` 同时进入编码器和 BCE。像素 0 对总损失的导数为

$$
\frac{\mathrm{d}\mathcal{J}}{\mathrm{d}x_0}
=\underbrace{-u_0}_{\text{目标值变化}}
+\underbrace{0.027242988513}_{\text{编码器路径}}
=-0.050838198238.
$$

若目标固定、只扰动编码器输入，导数会少掉 $-u_0$ 这一项。源码在 [`network_backward` 第 279-285 行](../src/ml/13_vae_shapes.c#L279-L285) 先写入目标路径 `-logit`，再在第 312-318 行叠加编码器路径。

## 13. 批量训练、裁剪与 Adam 的一次更新

一个训练批次有 8 张图。代码对每张图分别采样 $\boldsymbol{\epsilon}$，计算完整负 ELBO 梯度，再对梯度取批量平均：

$$
\mathbf{g}_{B}=\frac1{B}\sum_{s=1}^{B}\nabla_{\mathbf{w}}\mathcal{J}_s,
\qquad B=8,
$$

其中 $\mathbf{w}\in\mathbb{R}^{16900}$ 是全部网络参数，$s$ 是当前批次内的样本索引。然后计算全局二范数

$$
G=\sqrt{\sum_{i=0}^{16899}g_{B,i}^{2}},
\qquad
\mathbf{g}'_{B}=\begin{cases}
\mathbf{g}_{B},&G\leq5,\\
\frac{5}{G}\mathbf{g}_{B},&G>5.
\end{cases}
$$

裁剪只缩放梯度，不改变方向。`clip_gradient` 使用 `hypot` 累计范数，避免直接平方求和更容易溢出；发现非有限梯度时会在改动数组前返回失败。

Adam 的更新为

$$
\begin{aligned}
m_t&=0.9m_{t-1}+0.1g'_t,&
v_t&=0.999v_{t-1}+0.001(g'_t)^2,\\
\widehat m_t&=\frac{m_t}{1-0.9^t},&
\widehat v_t&=\frac{v_t}{1-0.999^t},\\
w_t&=w_{t-1}-0.003\frac{\widehat m_t}{\sqrt{\widehat v_t}+10^{-8}}.
\end{aligned}
$$

源码把 $\beta_1^t$、$\beta_2^t$ 分别保存在 `beta1_power`、`beta2_power` 中，每个批次乘一次衰减率。`adam_update` 先在局部副本中计算全部新参数及矩估计，确认有限后才一起写回；某个数值溢出不会留下半更新模型。

为单独观察这几步，可将上面的固定梯度检查样本视作一个大小为 1 的教学批次。该样本的裁剪前梯度范数为 $G=24.948471716593$，所以缩放系数为 $5/G=0.200413077675$。`OB[0]` 的梯度从 $0.419510385327$ 变成 $0.084075367440$。初始偏置及 Adam 矩估计均为零，第一步因此有

$$
m_1=0.008407536744,\qquad
v_1=0.000007068667,\qquad
\widehat m_1=0.084075367440,\qquad
\widehat v_1=0.007068667\;.
$$

代入学习率后，`OB[0]` 从 $0$ 更新到约 $-0.003$（实测 $-0.002999999643$）。这只是便于手算的单样本更新；实际训练先平均 8 张图的梯度，所以实际首批更新值不同。

## 14. 一次真实训练运行如何解读

入口 [`main`](../src/ml/13_vae_shapes.c#L693-L774) 固定生成 96 张训练图、48 张验证图和 48 张独立测试图，种子分别为 2025、2026、2027；网络初始化种子为 42。数据每三张依次生成圆、方、三角，中心和半径由确定性随机数变化。训练按数组顺序分成 12 个批次，每轮 96 次重参数采样；没有打乱数据顺序。随机噪声仍会逐样本更新，所有实验可在相同实现和浮点环境下复现。

每轮结束后，验证集使用重置为 5025 的固定噪声状态，每张图采 4 次；这让不同轮次在同一组噪声上比较。代码另以 $\boldsymbol{\epsilon}=\mathbf{0}$ 解码 $\boldsymbol{\mu}$，计算

$$
\operatorname{MSE}
=\frac1{256}\sum_{j=0}^{255}(x_j-\hat{x}_j)^2,
$$

它只用于可读的重建比较，不参与训练，也不代表随机后验解码的期望误差。检查点依据验证集平均负 ELBO 选择；训练结束时会恢复最佳轮次的完整参数副本。

本次执行 `./build/ml/13_vae_shapes` 的验证过程如下。表中均是每张验证图的平均值：

| 轮次 | 重建 NLL | KL | 负 ELBO | 均值重建 MSE |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 110.9751 | 10.9692 | 121.9443 | 0.12674 |
| 50 | 21.5318 | 2.1932 | 23.7249 | 0.02466 |
| 100 | 20.0317 | 2.3170 | 22.3487 | 0.02346 |
| 150 | 16.8251 | 3.0715 | 19.8967 | 0.01817 |
| 200 | 15.1284 | 3.3451 | 18.4735 | 0.01602 |
| 250 | 14.2033 | 3.5229 | 17.7263 | 0.01433 |

第 250 轮日志不是最终使用的模型：验证集最优检查点是第 243 轮，负 ELBO 为 17.5418。恢复该检查点后，独立测试集得到重建 NLL 14.5309、KL 3.3248、负 ELBO 17.8557、均值重建 MSE 0.01348。训练集平均图在测试集上的 MSE 为 0.03990，因此当前模型比“永远输出平均图”更能重建这些独立合成图；这个比较仍然只覆盖本例的三类合成图。

## 15. 数值测试能证明什么，以及模型边界

`test_gradients` 用中心差分

$$
g_i^{\mathrm{numeric}}
=\frac{\mathcal{J}(v_i+10^{-5})-\mathcal{J}(v_i-10^{-5})}{2\cdot10^{-5}}
$$

逐一核对全部 16900 个参数、256 个输入和 2 个噪声分量。这里 $v_i$ 表示当前被微扰的一个标量；$g_i^{\mathrm{analytic}}$ 是反向传播算出的对应导数。以输出偏置 `OB[0]` 为例，固定 $\boldsymbol{\epsilon}$ 后，实测正向损失为 $180.267904311076$，负向损失为 $180.267895920869$，所以数值导数为 $0.419510385541$，解析导数为 $0.419510385327$。比较条件是

$$
|g_i^{\mathrm{numeric}}-g_i^{\mathrm{analytic}}|
<2\cdot10^{-6}+2\cdot10^{-4}|g_i^{\mathrm{numeric}}|.
$$

正负扰动始终复用同一个 $\boldsymbol{\epsilon}$，否则两次前向的差异会混入重新采样的随机误差。边界测试另验证零均值/单位方差时 KL 为零、均值非零的解析 KL、$\ell$ 上下界、极端 logit 的有限损失、随机序列可复现、梯度裁剪以及 Adam 遇到非有限值时不部分更新，见 [`test_boundaries`](../src/ml/13_vae_shapes.c#L540-L579)。

本次重新执行的 `make test-vae` 与 `make sanitize-vae` 均通过；后者使用 AddressSanitizer 和 UndefinedBehaviorSanitizer。测试证明的是当前固定结构、合成数据、数值梯度和内存边界符合预期，不证明模型能泛化到真实图片。该模型是全连接 MLP，没有卷积层、标签条件或真实图像数据；潜变量只有 2 维，输出是 256 个独立 Bernoulli 概率。因此模糊轮廓、混合形状或某些先验样本不清晰，都是此教学规模下可能出现的结果。
