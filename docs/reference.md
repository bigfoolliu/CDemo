# 数学公式与符号读法参考

这份参考归纳 `docs/` 中公式专题反复使用的数学记号，帮助读出公式并辨认符号。它按符号和主题总结，不逐条复制专题页的长推导；具体变量仍以公式附近的定义和对应代码为准。

## 先学会拆读公式

| 写法 | 读法 | 如何理解 |
| --- | --- | --- |
| $x_{i,j}$ | “x 下标 i、j” | 通常表示第 $i$ 个对象的第 $j$ 个分量；下标是编号，不是乘法。 |
| $W^{(l)}$ | “W 上标括号 l” | 常表示第 $l$ 层的权重；括号里的上标是层编号。 |
| $x^2$ | “x 的平方” | 这里的上标才表示乘方。 |
| $W^{\mathsf T}$ | “W 的转置” | 上标 $\mathsf T$ 表示交换矩阵的行和列。 |
| $Q^*$、$y^\star$ | “Q 星号”、“y 星号” | 星号可能表示最优值或目标值，具体含义由所在教程定义。 |
| $\hat{x}$、$\tilde{x}$、$\bar{x}$ | “x 帽子”、“x 波浪”、“x 横线” | 常分别表示估计/标准化值、变换/扰动值、平均值；不是跨主题固定的规则。 |
| $\mathbf{x}$、$\boldsymbol{\theta}$ | “粗体 x”、“粗体 theta” | 通常表示向量或一组参数；普通 $x$、$\theta$ 常表示其中一个数。 |
| $A\in\mathbb{R}^{m\times n}$ | “A 属于 m 乘 n 维实数矩阵” | $\mathbb{R}$ 是实数集合，$m$ 是行数，$n$ 是列数。 |

文字形式的上下标也很常见。比如 $W^{\mathrm{head}}_{j,d}$ 读作“W 上标 head、下标 j 和 d”；`head` 是分类头的标签，不是乘方。$g^{\mathrm{num}}$ 中的 `num` 表示数值估计，$g^{\mathrm{analytic}}$ 中的 `analytic` 表示解析计算。

下标一般用于标记样本、时间、类别、层、特征或求和位置，但具体角色会变化。例如字符 RNN 和 Transformer 常用 $t$ 表示时间位置；Adam 教程中的 $t$ 或 $n$ 也可能表示参数更新次数。遇到 $i,j,k$ 时，先确认公式前后定义了它们遍历什么。

## 常见英文字母

| 记号 | 常见读法与用途 | 阅读提醒 |
| --- | --- | --- |
| $x$、$\mathbf{x}$、$X$ | 输入、特征向量或数据矩阵 | 小写常指一个样本或向量，大写常指矩阵/整批数据；以专题定义为准。 |
| $y$、$\hat{y}$ | 标签/目标、模型预测 | 分类中 $y$ 常是类别；回归中 $y$ 常是数值目标。帽子常标记预测。 |
| $w$、$W$、$b$ | 权重、权重矩阵、偏置 | $W$ 连接多组神经元；$b$ 通常加在线性变换之后。 |
| $z$、$p$ | 分数或激活前数值、概率 | 分类常先由 $z$ 得到 $p$；但 Transformer 的残差张量也会使用 $z$。 |
| $L$、$\ell$、$J$、$\mathcal{L}$ | 损失或优化目标 | 常用 $\ell_i$ 表示单个样本损失，用 $L$ 或 $J$ 表示平均损失/总目标；大小写并无统一强制规则。 |
| $i,j,k$ | 求和下标或数组编号 | 它们只是占位编号；每个公式都要看清遍历范围。 |
| $N,n,d$ | 样本数、某种规模、特征维度或距离 | $N$ 常表示样本总数；$d$ 既可能表示维度，也可能表示距离，需看上下文。 |
| $t,h,l$ | 时间步、注意力头、网络层 | 同一个字母在其他公式也可能表示别的量。 |
| $h_t,c_t$ | “h/c 下标 t” | 循环网络的隐藏状态和记忆状态；$t$ 是序列位置。 |
| $Q,K,V$ | “Q、K、V” | Transformer 中是查询、键、值；$Q(s,a)$ 在强化学习中则是动作价值函数。 |
| $G,D$ | “G、D” | GAN 中通常是生成器和判别器；强化学习的 $G_t$ 也可能表示累计回报。 |

### 按模型辨认常见变量

神经网络的层公式常写成：

$$
z^{(l)}=W^{(l)}a^{(l-1)}+b^{(l)},\qquad
a^{(l)}=\phi\!\left(z^{(l)}\right)
$$

读作“第 $l$ 层先把上一层输出乘权重并加偏置，得到激活前值 $z^{(l)}$；再应用激活函数 $\phi$，得到本层输出 $a^{(l)}$”。这里括号上标 $(l)$ 是层编号。教程也常用 $x$ 直接表示网络输入，用 $h$ 表示隐藏状态或隐藏层输出；$g$ 常表示梯度，$\delta$ 常表示反向传播中的误差信号。

Adam 公式里的 $m$、$v$ 不是模型输入，而是优化器保存的历史量：

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\qquad
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2
$$

$m_t$ 读作“第 $t$ 次更新的一阶矩”，近似保存梯度的移动平均；$v_t$ 是梯度平方的移动平均。带帽子的 $\hat m_t$、$\hat v_t$ 是偏差修正后的量，$\beta_1$、$\beta_2$ 是衰减系数，$\eta$ 是学习率，$\epsilon$ 通常用于数值稳定。不同教程可能用 $t$ 或 $n$ 表示更新次数；序列模型里的 $t$ 则通常是字符或时间位置。

#### CNN 卷积与 LayerNorm

CNN 常把张量形状写成 $N\times C\times H\times W$：依次读作“批次、通道、高度、宽度”。本项目 CNN 案例使用单通道输入，其卷积计算可读作：

$$
Z_{q,r,s}=b_q^{\mathrm{conv}}+
\sum_{a=0}^{K_h-1}\sum_{b=0}^{K_w-1}K_{q,a,b}X_{r+a,s+b},\qquad
A_{q,r,s}=\max(0,Z_{q,r,s})
$$

$q$ 是输出通道，$(r,s)$ 是输出像素位置，$(a,b)$ 是卷积核内部位置；$K_h,K_w$ 是卷积核的高和宽，$K_{q,a,b}$ 是对应权重，$Z$ 是激活前值，$A$ 是 ReLU 后的值。式中的求和读作“把窗口里每个像素乘对应权重，再全部相加”。符号 $K$ 在 K 近邻里表示邻居数，在 Transformer 里表示键（Key），在 CNN 里表示卷积核，须看上下文判断。

LayerNorm 对同一序列位置的特征做归一化，再用可学习参数调整：

$$
\hat{x}_{t,i}=\frac{x_{t,i}-\mu_t}{\sqrt{\sigma_t^2+\epsilon}},\qquad
u_{t,i}=\gamma_i\hat{x}_{t,i}+\beta_i
$$

$t$ 是位置，$i$ 是特征编号；$\mu_t$、$\sigma_t^2$ 是该位置所有特征的均值和方差。$\gamma_i$ 缩放标准化结果，$\beta_i$ 平移结果；这里的 $\beta_i$ 是模型参数，与 Adam 的衰减系数 $\beta_1,\beta_2$ 不同。

同一字母在项目中有几组容易混淆的含义：

| 记号 | 常见含义 | 区分方法 |
| --- | --- | --- |
| $z$ | 神经元激活前值或分类 logit；VAE 潜变量；GRU 更新门 | 看它是标量/向量分数、随机潜变量，还是位于 $h_{t+1}$ 状态混合式中。 |
| $h$、$H$ | 隐藏状态/隐藏宽度；CNN 图像高度；注意力头数 | 看下标时间 $t$、张量形状位置，或定义表中的维度。 |
| $c$、$C$ | 类别编号；CNN 通道数；RNN 记忆状态 | 小写常是单个类别或状态，大写常是维度规模，但以原文定义为准。 |
| $r_t$ | GRU 重置门；DQN 即时奖励 | 看公式是否在 sigmoid 门控式中，或出现在回报目标中。 |
| $Q$ | Transformer 查询（Query）；强化学习动作价值函数 | 是否带 $K,V$ 参数表示注意力；是否写成 $Q(s,a)$ 表示动作价值。 |
| $m$、$v$ | Adam 的一、二阶矩；Transformer 中间表示、值向量 | 看它们是否带优化步下标并与 $\beta_1,\beta_2$ 一起出现。 |
| $\beta$ | Adam 衰减系数；LayerNorm 平移参数；VAE 的 KL 权重 | 下标（如 $\beta_1$、$\beta_{\mathrm{KL}}$）和所在公式会说明用途。 |

## 希腊字母怎么读

希腊字母的名字可以照下面读。右栏列出它们在本项目教程中的常见含义；这些含义只作导航，不是统一定义。

| 符号 | 中文近似读法 | 常见含义 |
| --- | --- | --- |
| $\alpha$ | 阿尔法 | 学习率、朴素贝叶斯平滑强度、初始化尺度。 |
| $\beta$ | 贝塔 | Adam 的移动平均系数；LayerNorm 中也用作可学习平移量。 |
| $\gamma$ | 伽马 | DQN 的未来奖励折扣率；归一化层中也可表示可学习缩放量。 |
| $\delta$、$\Delta$ | 小写/大写德尔塔 | $\delta$ 常表示误差或梯度；$\Delta$ 常表示变化量。 |
| $\epsilon$、$\varepsilon$ | 艾普西龙 | 数值稳定常量、有限差分扰动量；DQN 中也可表示随机探索率。 |
| $\eta$ | 伊塔 | 常表示学习率。 |
| $\theta$、$\vartheta$ | 西塔 | $\theta$ 常表示模型参数；$\vartheta$ 在随机采样公式中可表示角度。 |
| $\kappa$ | 卡帕 | 参数或类别的编号，依公式定义。 |
| $\lambda$ | 兰姆达 | L2 正则化强度；PCA 中也可表示特征值。 |
| $\mu$ | 缪 | 均值或聚类中心。 |
| $\pi$ | 派 | 圆周率。 |
| $\rho$ | 柔 | 半径、密度或残差量，依主题而异。 |
| $\sigma$ | 西格玛 | 标准差，也可作为 sigmoid 函数的名称；$\sigma^2$ 通常表示方差。 |
| $\tau$ | 陶 | 阈值、裁剪上限或某些教程中的索引。 |
| $\phi$、$\Phi$ | 小写/大写 fai | 标准正态密度与累积分布函数；其他主题可能用作函数或势函数。 |
| $\chi$ | 凯（有时读 chi） | 常作指示函数或掩码，例如标记 ReLU 是否通过。 |
| $\xi$ | 克西（近似音 ksi） | 随机变量或坐标。 |
| $\upsilon$ | 宇普西龙 | 例如坐标记号。 |
| $\psi$ | 普西 | 例如 GAN 判别器的参数。 |
| $\zeta$ | 泽塔 | 例如 Transformer 中为避免重名而另记的 logit。 |

大小写也有区别。例如 $\delta$ 和 $\Delta$ 是不同字母；$\phi$ 与 $\Phi$ 在正态分布中分别代表密度函数和累积分布函数。

## 运算符和关系符

| 符号 | 怎么读 | 含义或例子 |
| --- | --- | --- |
| $\sum_{i=0}^{N-1}$ | “对 i 从 0 到 N 减 1 求和” | 把所有样本或分量的值加起来。 |
| $\prod_{i=1}^{N}$ | “对 i 从 1 到 N 连乘” | 把一组因子相乘，常见于概率联合分布。 |
| $\frac{\partial L}{\partial w}$ | “L 对 w 的偏导数” | 只改变 $w$ 时，损失 $L$ 的局部变化率。 |
| $\nabla_{\theta}L$ | “L 对 theta 的梯度” | 损失对所有参数分量的偏导数组成的向量。 |
| $\odot$ | “逐元素相乘” | 两个同形状向量对应位置相乘。 |
| $\mathbf{x}^{\mathsf T}\mathbf{y}$、$\mathbf{x}\cdot\mathbf{y}$ | “内积”或“点积” | 对应分量相乘后求和。 |
| $\lVert\mathbf{x}\rVert_2$ | “x 的二范数” | 向量的欧几里得长度；平方后为各分量平方和。 |
| $\max$、$\min$、$\arg\max$、$\arg\min$ | 最大值、最小值、最大值对应的输入、最小值对应的输入 | `argmax` 返回让函数取最大值的输入或编号，不是最大值本身。 |
| $\exp(x)$、$\ln(x)$、$\log(x)$ | 指数函数、自然对数、对数 | 文档若未另行说明，机器学习中的 $\log$ 通常按自然对数处理。 |
| $\mathbf{1}[A]$ | “A 成立时为一的指示函数” | 条件为真时取 $1$，否则取 $0$。 |
| $\mathbb{P}(A)$、$\mathbb{E}[X]$ | A 的概率、X 的期望 | 期望是按可能结果的概率加权平均。 |
| $P(A\mid B)$ | “给定 B 时 A 的条件概率” | 竖线 $\mid$ 读作“在……条件下”。 |
| $x\sim\mathcal{N}(\mu,\sigma^2)$ | “x 服从均值 mu、方差 sigma 平方的正态分布” | $\mathcal{N}$ 表示正态分布。 |
| $\operatorname{Var}(X)$、$\operatorname{Cov}(X,Y)$ | 方差、协方差 | 方差描述单个量的分散程度，协方差描述两个量共同变化的方向。 |
| $D_{\mathrm{KL}}(q\mathbin{\|}p)$ | “q 与 p 之间的 KL 散度” | 比较两个概率分布；顺序有意义，通常不对称。 |
| $\in$、$\notin$、$\subseteq$ | 属于、不属于、是子集 | 用于说明数值范围、类别集合和索引集合。 |
| $\le$、$\ge$、$\ne$、$\approx$ | 小于等于、大于等于、不等于、约等于 | 约等号通常表示数值经过了舍入或近似。 |
| $\bmod$、$\lfloor x\rfloor$、$\lvert x\rvert$ | 取模、向下取整、绝对值 | 常用于伪随机数、索引和距离计算。 |
| $\rightarrow$、$\leftarrow$、$\Longrightarrow$、$\Longleftrightarrow$ | 映射到/趋向、从……更新为、推出、等价 | 梯度下降常写 $\theta\leftarrow\theta-\eta\nabla_\theta L$，表示用新值替换旧参数。 |
| $\propto$ | 正比于 | 一边等于另一边乘上与当前变量无关的常数或归一化因子。 |
| $\operatorname{ReLU}(x)$、$\sigma(x)$、$\tanh(x)$、$\operatorname{GELU}(x)$ | ReLU、sigmoid、双曲正切、GELU | 神经网络的激活函数；例如 $\operatorname{ReLU}(x)=\max(0,x)$。 |
| $\operatorname{softmax}(\mathbf{z})$ | “z 向量的 softmax” | 把一组分数变成总和为 $1$ 的概率。 |

## 常见公式该怎么念

### 线性模型与损失

$$
\hat{y}=\mathbf{w}^{\mathsf T}\mathbf{x}+b,
\qquad
L=\frac{1}{N}\sum_{i=1}^{N}(y_i-\hat{y}_i)^2
$$

第一式读作“预测值等于权重向量与输入向量的内积，再加偏置”。第二式读作“平均损失等于每个样本的真实值减预测值之差的平方，再对 $N$ 个样本取平均”。这里 $\mathbf{x}$ 是输入、$\mathbf{w}$ 是权重、$b$ 是偏置、$y_i$ 是目标值、$N$ 是样本数。平方误差是线性回归教程中的例子；其他分类任务会使用交叉熵或 hinge 损失。

### 概率与分类

$$
p_k=\frac{\exp(z_k)}{\sum_j\exp(z_j)},
\qquad
\ell=-\ln p_y
$$

第一式读作“类别 $k$ 的概率等于它的分数指数，除以所有类别分数指数的总和”；第二式读作“目标类别概率的负自然对数”。$z_k$ 是类别分数，$p_k$ 是概率，$y$ 是目标类别。实现中常先从所有 $z_k$ 减去最大值，避免指数溢出；这不改变归一化后的概率。

### 梯度下降与正则化

$$
\boldsymbol{\theta}_{\mathrm{new}}
=\boldsymbol{\theta}_{\mathrm{old}}-\eta\nabla_{\boldsymbol{\theta}}L,
\qquad
J=L_{\mathrm{data}}+\frac{\lambda}{2}\lVert\mathbf{w}\rVert_2^2
$$

第一式读作“新参数等于旧参数减去学习率乘损失梯度”，所以沿负梯度方向更新。第二式读作“优化目标等于数据损失加上权重平方和的惩罚”；$\lambda$ 控制惩罚强度，$\mathbf{w}$ 的范数项不会自动包含偏置，是否惩罚偏置要看实现。

### 经典机器学习公式

| 主题 | 公式 | 读法要点 |
| --- | --- | --- |
| K 近邻 / K-means | $d(\mathbf{x},\mathbf{z})=\sqrt{\sum_j(x_j-z_j)^2}$ | 对每个特征求差、平方、相加再开方；K-means 再将样本分给距离最近的中心 $\boldsymbol{\mu}_k$。 |
| 朴素贝叶斯 | $P(c\mid\mathbf{x})\propto P(c)\prod_jP(x_j\mid c)$ | 后验分数正比于类别先验乘以每个特征的条件概率；朴素假设是给定类别后特征条件独立。 |
| 决策树 | $G=1-\sum_k p_k^2$ | $p_k$ 是节点内第 $k$ 类的比例；$G$ 越小，节点越纯。随机森林会对树的预测再投票或平均。 |
| PCA | $C\mathbf{u}=\lambda\mathbf{u}$ | 协方差矩阵 $C$ 乘方向 $\mathbf{u}$，等于特征值 $\lambda$ 乘该方向；特征值表示沿该方向的方差。 |
| 线性 SVM | $\ell_i=\max(0,1-y_i f(\mathbf{x}_i))$ | $y_i\in\{-1,+1\}$ 是类别，$f(\mathbf{x}_i)$ 是分类分数；正确且间隔足够大时损失为零。 |
| 交叉验证 | $\bar{J}=\frac{1}{K}\sum_{k=1}^{K}J_k$ | 将 $K$ 折的验证分数 $J_k$ 求平均，再用平均表现选择配置。 |

损失名称也可以当成公式的“标题”：MSE 是均方误差，BCE 是二元交叉熵，CE 是多类交叉熵，KL 是概率分布之间的散度，hinge 是间隔损失，ELBO 是 VAE 常用的证据下界目标。具体正负号、平均方式和正则项以相应教程的定义为准。

### 循环网络与生成模型

$$
h_{t+1}=\tanh(W_xx_t+W_hh_t+b_h)
$$

读作“下一时刻的隐藏状态，等于输入权重乘当前输入，加上循环权重乘旧状态，再加偏置，最后逐分量应用 tanh”。$x_t$ 是当前输入，$h_t$ 是当前隐藏状态，$W_x,W_h$ 是两组权重。GRU 会再引入更新门 $z_t$、重置门 $r_t$ 和候选状态 $n_t$；本项目的状态混合式为：

$$
h_{t+1}=z_t\odot h_t+(1-z_t)\odot n_t
$$

这里 $z_t$ 是保留旧状态的比例，$r_t$ 控制候选状态读取旧状态的程度；$\odot$ 表示逐元素相乘。

$$
\mathbf{z}=\boldsymbol{\mu}(\mathbf{x})
+\exp\!\left(\tfrac12\boldsymbol{\ell}(\mathbf{x})\right)\odot\boldsymbol{\epsilon},
\qquad
\boldsymbol{\epsilon}\sim\mathcal{N}(\mathbf{0},I)
$$

这是 VAE 的重参数化式，读作“潜变量样本等于均值，加上标准差逐元素乘标准正态噪声”。$\boldsymbol{\ell}$ 是对数方差，所以 $\exp(\boldsymbol{\ell}/2)$ 是标准差；$I$ 是单位协方差矩阵。这样写可以把随机噪声 $\boldsymbol{\epsilon}$ 与可学习的均值、方差参数分开。

GAN 的生成步骤可写为 $\mathbf{y}=G_{\theta}(\mathbf{z})$，判别器再计算 $D_{\psi}(\mathbf{y})$。读作“生成器 $G$ 用参数 $\theta$ 把潜变量 $\mathbf{z}$ 变成样本 $\mathbf{y}$；判别器 $D$ 用参数 $\psi$ 估计样本为真的概率”。这里 $\mathbf{z}$ 是随机潜变量，不能和分类里的 logit $z$ 混为一谈。

ResNet 的捷径可以用 $H=\operatorname{ReLU}(S+F(S))$ 表示，读作“主分支 $F(S)$ 与直接传过来的 $S$ 相加，再通过 ReLU”。$S$ 是块输入，$F(S)$ 是卷积分支输出；相加让梯度也可以沿恒等路径直接回传。

### 注意力和强化学习

$$
\operatorname{Attention}(Q,K,V)
=\operatorname{softmax}\left(\frac{QK^{\mathsf T}}{\sqrt{d_k}}\right)V
$$

读作“先用查询 $Q$ 与键 $K$ 的转置做相似度，再除以键的特征宽度平方根，经过 softmax 变成权重，最后对值 $V$ 加权”。$d_k$ 是键向量维度；本项目的 Transformer 教程将每头的这个维度写作 $d_h$。因果掩码等限制由具体实现另行施加。

$$
y_t=
\begin{cases}
r_t,&\text{terminal},\\
r_t+\gamma\max_{a'}Q_{\theta^-}(x_{t+1},a'),&\text{otherwise}
\end{cases}
$$

读作“若本步终止，学习目标就是即时奖励；否则，再加上折扣后的下一状态最大动作价值”。$x_t$ 是观测，$a'$ 是下一步候选动作，$r_t$ 是即时奖励，$\gamma$ 是折扣率，$Q_{\theta^-}$ 是目标网络估计的动作价值。这个目标用于 Snake DQN；其他教程中的 $y_t$ 可能是普通监督标签。

## 按主题继续阅读

以下专题页包含完整推导、变量定义和数值例子。读法参考用于认符号，具体算法仍应以对应页面和 C 实现为准。

- 经典机器学习：[线性回归](linear_regression_analysis.md)、[逻辑回归](logistic_regression_analysis.md)、[K 近邻](knn_analysis.md)、[K-means](kmeans_analysis.md)、[朴素贝叶斯](naive_bayes.md)、[决策树](decision_tree.md)、[随机森林](random_forest.md)、[PCA](pca.md)、[线性 SVM](linear_svm.md)、[交叉验证](cross_validation.md)、[正则化](regularization.md)。
- 神经网络与深度学习：[数值基础](numeric_foundations.md)、[XOR MLP](xor_mlp.md)、[二维 MLP](mlp_2d.md)、[CNN](cnn_shapes.md)、[去噪自编码器](denoising_autoencoder.md)、[字符 RNN](char_rnn.md)、[GRU](gru_memory.md)、[Transformer](tiny_transformer.md)、[VAE](vae.md)、[GAN](gan_2d.md)、[ResNet](resnet.md)。
- 强化学习：[Snake DQN](snake_ai.md)。机器学习案例总览见[机器学习学习指南](machine_learning.md)，阶段路线见[深度学习学习计划](deep_learning_plan.md)。
