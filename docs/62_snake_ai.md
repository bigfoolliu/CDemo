# 纯 C 深度强化学习贪吃蛇

这个示例展示一个模型如何从随机动作出发，通过奖励和神经网络更新学习贪吃蛇。环境、训练器、反向传播、Adam 和模型文件全部使用 C 实现；不依赖 Python 训练或外部机器学习框架。Python 只用于可选的终端集成测试。

![贪吃蛇 DQN 从状态到经验回放的学习循环](images/62_snake_ai.svg)

当前状态产生动作与奖励，并形成一条经验；训练从经验池抽样更新网络，目标网络用于构造较稳定的目标值。

## 先从一局游戏理解强化学习

可以把 AI 想成一个反复练习的玩家。每走一步，游戏给它一份简化的局面描述；它从“直行、左转、右转”中选一个动作；游戏执行动作，返回新局面与奖励。训练器保存这次经历，再调整神经网络，让它以后更准确地估计每个动作的长期收益。

```text
游戏局面 Game ──观察──> 11 个数字 ──网络──> 3 个 Q 值 ──选动作──> 游戏执行
      ↑                                                        │
      └────────────────── 新局面、奖励、是否结束 ────────────────┘
                        │
                        └── 存入经验池 → 抽样学习 → 更新网络参数
```

这里的几个词有明确含义：

| 术语 | 在本项目中是什么 | 用来解决什么问题 |
| --- | --- | --- |
| 环境 | `Game` 和 `snake_act` | 按同一套规则移动、碰撞、吃食物、给奖励 |
| 观测 | `snake_observe` 生成的 11 个 `double` | 把棋盘转成网络可处理的数字 |
| 动作 | 编号 0、1、2 | 让网络只需在三个合法转向中选择 |
| 策略 | 根据 Q 值选动作的方法 | 决定下一步怎么走 |
| 奖励 | `SnakeResult.reward` | 提供这一步的反馈，未来收益由学习器继续估计 |
| 训练 | 调整 `Network.p` 中的参数 | 让预测 Q 值逐渐接近学习目标 |

普通监督学习有现成的“正确答案”；贪吃蛇没有一张记录每个局面该怎么走的答案表。DQN（Deep Q-Network）把 Q 学习的动作价值估计交给神经网络实现，属于**强化学习**：先尝试动作，观察后果，再利用奖励学习。网络输出的是**动作价值** $Q(s,a)$，即处于观测 $s$ 时执行动作 $a$ 后，预期能得到多少折扣后的累计奖励。它不直接预测“这步是否正确”，也不输出三个动作的概率。

第一次阅读建议按“规则 → 观测与奖励 → 网络前向计算 → 一条经验的学习 → 完整训练循环 → 评估”顺序进行。每读完一层，就尝试用自己的话解释它的输入、输出和修改了什么状态。

## 运行步骤

```sh
make snake-ai
./build/games/snake_ai train --seconds 180 --seed 42
```

程序输出独立的运行目录，例如 `sessions/snake-ai/run-XXXXXX/`。用实际目录替换下面的路径：

```sh
./build/games/snake_ai eval --model sessions/snake-ai/run-XXXXXX/best.model
./build/games/snake_ai play --model sessions/snake-ai/run-XXXXXX/best.model
less logs/snake-ai/run-XXXXXX/train.log
```

训练默认最多 180 秒或 20000 局，任一到达即停止。若本机较快、希望跑满三分钟，可提高局数上限：

```sh
./build/games/snake_ai train --seconds 180 --episodes 100000 --seed 42
```

`--steps N` 可以限制总环境步数，方便短试跑和可重复性检查；时间、局数和总步数限制同时生效。训练过程中 Ctrl+C 会停止采样并保存结果，随后完成最终评估。初始验证和最终评估不计入训练计时，周期验证计入训练计时。

`eval` 默认用固定测试种子 2000001、100 局和每局 4000 步上限；`--seed` 可以改变评估种子起点。`play` 用 200 毫秒间隔观看模型自动控制，不允许人工转向，按 Q 或 Ctrl+C 退出。死亡后自动重开。演示使用独立的 `ai_scores.log`，不影响人工历史。

初次学习可以先跑短实验，观察目录和日志的格式；短实验通常不足以学会高分：

```sh
./build/games/snake_ai train --seconds 30 --episodes 100000 --steps 2000 --seed 42
```

三个预算条件同时生效，先到达哪个就停止采样。程序还会做训练前验证和训练后评估，因此总运行时间可能超过 `--seconds`。不要把短试跑的分数当作模型能力的结论。

## 阅读源码前需要的 C 基础

项目没有机器学习库，数据结构就是普通 C 数组和结构体。先认清下列写法，后面的公式就能和代码对应起来：

```c
Game game = snake_reset(42);     /* 结构体：包含蛇身、方向、食物和随机状态。 */
double state[FEATURES];          /* 栈上的 11 个输入数字。 */
snake_observe(&game, state);      /* &game 传入地址；函数写入 state 数组。 */
SnakeResult result = snake_act(&game, 0); /* game 会被修改，result 按值返回。 */
```

`game.body[0]` 是蛇头，`game.length` 是长度；若变量是 `Game *game`，用 `game->length` 访问成员。`const Game *` 表示函数只读取游戏，`Game *` 表示可能修改游戏。`size_t` 用于数组长度和索引，`double` 用于特征、奖励、权重与梯度。`snake_env.h` 和 `snake_ai_network.h` 中的 `static inline` 函数写在头文件里，使人工游戏、AI 和测试可以各自包含同一份实现。

`Network.p` 是连续的 `double` 参数数组，`Forward` 保存一次前向计算的中间结果，`Transition` 保存一次动作前后的经验。`Learner` 含一万个经验，体积较大；`train` 用 `calloc` 在堆上创建它，并在结束时 `free`。阅读时分清“只是读取”“按值复制”和“通过指针修改”，就能追踪每一步的数据流。

## 数学预备：从数字到变化率

本节只引入后续推导需要的记号。公式用 LaTeX 排版；先把它们理解成对 C 数组和循环的简洁写法。

| 记号 | 含义 | 代码对应 |
| --- | --- | --- |
| $x_t, x_{t+1}$ | 当前和下一步的 11 项观测 | `state`、`next` |
| $a_t, r_t$ | 当前动作和得到的奖励 | `action`、`reward` |
| $Q_\theta(x,a)$ | 网络对动作长期价值的预测 | `Forward.q[a]` |
| $\theta, \theta^-$ | 在线网络和目标网络参数 | 两个 `Network.p` 数组 |
| $\gamma, \varepsilon$ | 未来奖励折扣率、随机探索率 | `0.95`、训练中的探索率 |
| $L, \nabla_\theta L$ | 损失和所有参数的梯度 | 损失值、梯度数组 |
| $s,a$ | 公式中泛指的状态与动作；本项目分别写作完整局面 $g$、观测 $x$ 和动作编号 $a$ | 游戏局面、11 项状态、动作编号 |
| $g_t,\phi$ | 第 $t$ 步的完整游戏局面、把局面编码成观测的函数 | 游戏环境、观测编码函数 |
| $r_{\text{base}},\Phi$ | 游戏原始奖励、根据局面计算的势函数 | 基础得分、势函数 |
| $G_t,\pi,\mathbb{E}$ | 从第 $t$ 步开始的折扣回报、后续策略、对可能结果的概率加权平均 | 训练轨迹及其平均意义 |
| $W^{(l)},b^{(l)},z^{(l)},h^{(l)},l$ | 第 $l$ 层的权重、偏置、激活前结果、激活后输出和层编号 | 网络参数、前向计算缓存 |
| $y_t,e_t,d,\delta^{(l)}$ | TD 学习目标、预测误差、损失对所选 Q 值的导数、传回第 $l$ 层的误差 | 目标值、误差、导数和反传误差 |
| $B,N,S_i$ | 批次样本数、评估局数、第 $i$ 局得分 | 批次大小、评估局数、单局得分 |
| $g_i,\tilde g_i,m_i,v_i,\theta_i$ | 第 $i$ 个参数的平均梯度、裁剪后梯度、Adam 一阶/二阶平均、参数值 | 梯度数组、Adam 状态、网络参数 |

符号会按公式所在语境解释：$g_t$ 是游戏局面，$g_i$ 是梯度分量；几何部分的 $v$ 是方向向量，He 初始化中的 $v_x$ 是输入方差，Adam 部分的 $v_i$ 是梯度平方的滑动平均；距离公式中的 $d(H,F)$ 与反向传播中的导数 $d$ 也按上下文区分。矩阵或向量右上角的 $T$ 表示转置；$\lVert\cdot\rVert_2$ 表示欧几里得长度，例如 $(3,4)$ 的长度是 $\sqrt{3^2+4^2}=5$。下标 $t$ 通常表示环境步，Adam 公式会特别注明它表示更新次数；$i,j,k$ 是求和或数组下标，具体代表参数、神经元、输入特征还是评估局数，由对应公式旁的文字说明。$n$ 依上下文表示待平均的数目、网络层输入数或已走的环境步数。

**数、向量、矩阵。** 一个数叫标量；按固定顺序排列的一组数叫向量；行和列组成的数表叫矩阵。蛇的观测是 11 个数组成的向量。第一层权重是一个 $64\times11$ 矩阵，意思是 64 行、每行 11 个数。矩阵乘向量，就是每一行分别和向量做乘法再求和。

例如输入向量 $\boldsymbol{x}=(1,2)$，一行权重 $\boldsymbol{w}=(0.5,-1)$，偏置 $b=0.3$，则这一行的结果为：

$$
z = b + \sum_{j=1}^{2} w_j x_j
  = 0.3 + 0.5\times1 + (-1)\times2
  = -1.2.
$$

$\sum$ 表示把指定范围的项全部加起来。对应到 C，就是从 `j=0` 循环到输入长度，对 `weight[j] * input[j]` 累加。这里的 $\boldsymbol{w}$ 和 $\boldsymbol{x}$ 是向量，$b$、$z$ 是标量。

**比较、绝对值和平均数。** $|u|$ 表示数 $u$ 的绝对值；$\max(u,v)$ 取两个数中较大的一个；$\operatorname{argmax}_a Q(x,a)$ 返回让 $Q$ 最大的动作编号。若三局分数为 $2,4,0$，平均分为：

$$
\bar S = \frac{2+4+0}{3}=2.
$$

这里 $\bar S$ 表示平均分。分数平均值、最高分和奖励是不同统计量。$\frac{1}{n}\sum_{i=1}^{n}u_i$ 表示 $n$ 个数的平均值：$u_i$ 是第 $i$ 个数，$i$ 从 1 数到 $n$，$n$ 是总数。后文会用它平均一个训练批次的损失和梯度。

**概率和期望。** 概率从 0 到 1，分别表示不发生到必然发生。$X$ 表示随机变量，$\mathbb{E}[X]$ 是按各种结果的概率加权的平均值：若 $X$ 以一半概率取 $10$、一半概率取 $-10$，则 $\mathbb{E}[X]=0.5\times10+0.5\times(-10)=0$。期望不是某一局必然得到的分数。项目使用种子确定的伪随机数，固定种子便于复现实验；随机数取模可能带来极小偏差。

**导数和梯度。** 导数描述一个数变化一点时，另一个数大约变化多少。例如 $f(x)=x^2$，其导数为 $f'(x)=2x$；在 $x=3$ 附近，$x$ 增加很小的 $h$，$f$ 大约增加 $6h$。网络有很多参数，损失对每个参数各有一个偏导数；把这些偏导排成向量，就叫梯度 $\nabla_\theta L$。梯度给出损失上升最快的方向。对很小的参数变化 $\Delta\theta$，损失近似变化为：

$$
L(\theta+\Delta\theta)\approx L(\theta)+\nabla_\theta L^T\Delta\theta.
$$

若取 $\Delta\theta=-\eta\nabla_\theta L$ 且学习率 $\eta>0$，则右侧的变化量约为 $-\eta\lVert\nabla_\theta L\rVert_2^2\leq0$。这里 $\Delta\theta$ 是参数改变量，$\eta$ 是正的学习率，梯度中的每一项是损失对一个参数的偏导数，$\lVert\cdot\rVert_2$ 是这些偏导数组成向量的欧几里得长度。所以梯度下降沿负梯度更新；步长太大时，这个局部近似不再可靠。

**把梯度下降公式算一遍。** 取简单损失 $L(\theta)=\theta^2$，当前参数 $\theta=3$，则 $L=9$、梯度 $\nabla_\theta L=2\theta=6$。若学习率 $\eta=0.1$，更新量为 $\Delta\theta=-0.1\times6=-0.6$，新参数是 $2.4$，新损失为 $2.4^2=5.76$。一阶近似预测损失变化 $6\times(-0.6)=-3.6$；真实变化是 $5.76-9=-3.24$。两者接近但不相同，因为泰勒公式省略了二阶及更高阶项。

**链式法则。** 如果一个结果由多步计算得到，就逐步相乘传递变化率。令 $z=wx+b$，损失为 $L(z)$，那么：

$$
\frac{\partial L}{\partial w}
= \frac{\partial L}{\partial z}\frac{\partial z}{\partial w}
= \frac{\partial L}{\partial z}x.
$$

反向传播就是把这条规则从网络输出开始，按层反向应用。下面先介绍项目中的游戏状态、特征和代码结构，再在“从零推导训练过程与网络设计”中逐步推导每个数学量。

例如令 $x=2$、$b=1$、$w=3$，则 $z=wx+b=7$。若 $L(z)=z^2$，那么 $\partial L/\partial z=2z=14$，且 $\partial z/\partial w=x=2$，所以：

$$
\frac{\partial L}{\partial w}=14\times2=28.
$$

把 $w$ 从 3 改成 3.001，$z$ 从 7 变成 7.002，损失从 49 变成约 49.028004；损失约增加 0.028，和 $28\times0.001=0.028$ 一致。

## 学习过程：环境、状态、动作和奖励

### 环境规则

共享环境 `snake_env.h` 是人工游戏和 AI 的唯一规则实现。训练时直接调用单步函数，不刷新屏幕，也不等待。终端 API 和渲染在 `snake_terminal.h` 中，训练不需要交互终端。

棋盘宽 20、高 10，共 200 格。`Game.body[0]` 保存蛇头，其后的有效元素保存身体，初始长度为 3；每吃到一个食物，长度增加 1，屏幕得分等于 `length - 3`。`snake_reset(seed)` 固定初始蛇形和向右方向，并用种子放置食物。相同的环境种子与动作序列会得到可重复的食物序列。

游戏每一步的规则写在 `snake_env.h` 中：`snake_direction` 把相对动作转换为绝对方向；`step` 先检查墙和身体，再从尾到头移动数组；吃到食物时增长并在空格放新食物。没有吃食物时，旧尾格马上腾空，所以允许蛇头进入该格。`snake_collision` 用同一条件预测危险，避免“观测认为安全、执行时却判碰撞”。碰撞时 `step` 返回 `false`，不会把蛇身改成越界状态。

### 将棋盘编码为 11 个数字

模型的 11 个输入依次是：

| 索引 | 特征 |
| --- | --- |
| 0–2 | 直行、左转、右转是否马上碰撞 |
| 3–6 | 当前向左、右、上、下的方向编码 |
| 7–10 | 食物在蛇头左、右、上、下的方位 |

每项为 0 或 1。危险提示只描述下一步，允许进入即将离开的尾格。模型看不到整条蛇身、距离和可达区域：不同真实棋盘可能具有同一个输入，因此这是部分可观测的近似，并非完整的马尔可夫状态。“马尔可夫状态”指给定当前状态和动作后，预测未来不再需要更早的历史；这里的 11 个数字做不到这一点。

举例：蛇头在 `(10,5)`、朝右 `(1,0)`、食物在 `(13,3)`，且三个相对方向的下一格都安全。屏幕坐标的 `y` 向下增加，因此左转是向上，右转是向下。此时输入依次为：

```text
索引：  0 1 2 | 3 4 5 6 | 7 8 9 10
含义：  危险  | 当前方向 | 食物方位
数值：  0 0 0 | 0 1 0 0 | 0 1 1  0
```

食物同时在“右”和“上”，所以方位组有两个 1；方向组只有一个 1。输入不包含准确距离，也不记住之前走过的路。这个信息损失解释了模型即使学会短期避险，也可能把自己围死。

### 用三个 Q 值选择动作

输出为三个动作的 Q 值：0 直行、1 左转、2 右转。Q 值不是概率，而是对未来折扣奖励的估计。演示直接选最大 Q 值，不额外加入避险、寻路或满分规则。相同 Q 值时选择较小动作编号。

例如网络输出 `[2.1, 4.0, 1.5]`，贪心策略就选动作 1（左转）。即使左转会碰撞，代码也不会强行改选；网络必须从奖励中学会降低危险动作的 Q 值。训练时还会故意随机尝试动作，以便收集不同的结果。

### 奖励与结束条件

基础奖励为吃食物 +10、死亡 −10、普通移动 −0.01，填满棋盘额外 +100。为帮助早期探索，程序还加入基于蛇头到食物距离的势函数塑形。塑形项是 $\gamma\Phi(g_{t+1})-\Phi(g_t)$，其中 $\gamma=0.95$；吃到食物后会重新生成食物，下一局面的势使用新食物计算。完整推导和逐步计算的例子见后面的“从零推导训练过程与网络设计”。

游戏在死亡或满棋盘时真正终止；4000 步上限是时间截断，不是死亡。训练预算耗尽也不伪造死亡样本。当前棋盘共 200 格，初始蛇长 3，理论满分为 197。

## 网络与反向传播

网络结构为 `11 → 64 → 64 → 3`，共有 5123 个参数。两层隐藏层用 ReLU，输出为线性层；隐藏层权重采用 He 初始化，偏置为零。

### 前向传播：从输入算出 Q 值

前向计算可写为 $h^{(1)}=\operatorname{ReLU}(W^{(1)}x+b^{(1)})$、$h^{(2)}=\operatorname{ReLU}(W^{(2)}h^{(1)}+b^{(2)})$、$q=W^{(3)}h^{(2)}+b^{(3)}$。完整推导和各矩阵形状见后面的网络设计部分。

参数以输出行优先的平坦数组保存，便于看清 C 中矩阵索引。`Forward` 缓存输入和隐藏层输出，反向传播通过链式法则逐层累积权重及偏置的导数。学习只对实际执行动作的 Q 值计算损失。

一个全连接神经元先计算 $z=b+\sum_j w_jx_j$，再按需经过激活函数。$\operatorname{ReLU}(z)=\max(0,z)$：正数保留，负数变为零，给网络表示非线性关系的能力。最后一层不使用 ReLU，因为 Q 值可以为负。`network_forward` 依次调用三次 `dense`，把结果写入 `Forward.h1`、`h2` 和 `q`。

| 层 | 权重形状 | 偏置数 | 参数数 | 在 `Network.p` 中的起点 |
| --- | --- | ---: | ---: | --- |
| 输入 → 隐藏层 1 | 64 × 11 | 64 | 768 | `W1=0`，`B1=704` |
| 隐藏层 1 → 隐藏层 2 | 64 × 64 | 64 | 4160 | `W2=768`，`B2=4864` |
| 隐藏层 2 → 输出 | 3 × 64 | 3 | 195 | `W3=4928`，`B3=5120` |

所以总参数数为 $768+4160+195=5123$。以第一层第 `row` 个神经元、第 `column` 个输入为例，权重位置是 `p[W1 + row * FEATURES + column]`，偏置是 `p[B1 + row]`。这种一维数组布局就是 C 版矩阵；`static_assert` 在编译时检查各段没有重叠。`network_init` 用随机数初始化权重、把偏置清零，避免所有神经元始终学到相同的东西。

### 反向传播：从误差算出梯度

一次前向计算只是在**预测**，尚未修改参数。训练时 `network_gradient` 用误差的导数从输出层往回传：所选动作 $a$ 的输出偏置梯度为 $\partial L/\partial q_a$，对应权重的梯度为 $(\partial L/\partial q_a)h^{(2)}_i$。再通过该动作的权重把影响传回第二隐藏层，通过 $W^{(2)}$ 传回第一隐藏层；ReLU 输出为零的位置把梯度截断。本次损失只直接作用于执行过的动作，但共享隐藏层的权重变化也可能改变其他动作的预测。`Forward` 缓存的是反向传播所需的输入和隐藏层输出，不必重新跑网络。

可以先用只有一个隐藏神经元的缩小版手算，而不处理真实的 64 个神经元：若输入 $x=(1,0)$、权重 $w=(0.2,-0.5)$、偏置 $b=-0.1$，隐藏输出为 $h=\operatorname{ReLU}(0.2\times1-0.5\times0-0.1)=0.1$。若所选动作的输出权重为 $2$、偏置为 $0.3$，预测 Q 值是 $q=2\times0.1+0.3=0.5$。目标为 $y=1$ 时误差为 $e=q-y=-0.5$，Huber 导数也是 $-0.5$，这条输出权重的梯度就是 $(-0.5)\times0.1=-0.05$。真实代码只是把同样的加权、激活和链式求导扩展到三层数组。

梯度 $\partial L/\partial\theta$ 表示参数 $\theta$ 略微增加时损失 $L$ 会怎样变化。最基本的梯度下降是 $\theta_{\text{new}}=\theta_{\text{old}}-\eta\,\partial L/\partial\theta$，其中 $\eta$ 是学习率；梯度为负时，参数通常会增大。实际实现用 Adam 根据梯度历史调整每个参数的步幅，方向仍来自损失的导数。

### 批量学习与 Adam 更新

`learn_batch` 对 32 条样本把梯度累加到清零的 `Network gradient`；`adam_update` 先除以 32 取平均，再把整体梯度范数限制到 10。Adam 为每个参数保存一阶、二阶滑动平均，使用 $\beta_1=0.9$、$\beta_2=0.999$、学习率 $\eta=0.001$，并对初期从零开始的平均值做偏差校正。直观地说，损失的导数给出调整方向，Adam 根据最近的梯度历史决定每个参数的实际步幅。

源码 `snake_ai_network.h` 可按 `network_forward → network_gradient → adam_update` 顺序阅读。测试对所有参数做有限差分，核对解析梯度，而不是仅观察损失是否下降。

## DQN 为什么需要两张网络和经验池

训练器维护在线网络与目标网络。在线网络选动作和学习；目标网络提供下一状态估值，每 500 次参数更新复制在线参数，避免目标随每一步更新剧烈变化。

“未来折扣奖励”可写成 $G_t=\sum_{k=0}^{\infty}\gamma^k r_{t+k}$，其中 $G_t$ 是从当前步开始的累计回报，$k$ 是向未来的步数偏移，$r_{t+k}$ 是对应的单步奖励，$\gamma=0.95$ 是折扣率，越远的奖励权重越小。理想的 Q 值应满足贝尔曼最优性关系：当前价值约等于这一步奖励，加上下一步最有利动作的折扣价值。DQN 不知道真正的未来，只能用网络当前的估计构造一个**时序差分目标**，然后不断修正预测。在线网络与目标网络刚开始相同；之后在线网络频繁变化，目标网络每 500 次参数更新才同步一次，从而减轻“用变化中的答案追变化中的预测”造成的不稳定。

经验池保存 `(state, action, reward, next_state, terminated)`，容量为 10000，满后循环覆盖旧经验。预热 1000 步后，每四个环境步抽取一个 32 条样本的批次，有放回均匀采样。环境、探索动作和回放采样分别使用独立的 xorshift32 随机流。

每一步产生的 `Transition` 会按值复制到 `Replay.data`；`cursor` 指向下一格，写到末尾后绕回开头。回放打散连续游戏步之间的强相关性，也让旧经验可以被多次学习。它不能让样本变成真正独立的数据，也不保证池中各种局面数量均衡。

$$
y_t=
\begin{cases}
r_t, & \text{真正终止},\\
r_t+0.95\max_{a'}Q_{\theta^-}(x_{t+1},a'), & \text{其他情况},
\end{cases}
\qquad
e_t=Q_\theta(x_t,a_t)-y_t.
$$

达到时间上限时继续 bootstrap（自举），即用下一状态的 Q 估计补上尚未实际观察到的未来奖励。目标作为常数，不沿目标网络反向传播。Huber 损失在小误差时为平方项，大误差时为线性项；批量平均梯度的整体范数裁剪到 10 后，用学习率 0.001 的 Adam 更新。

用一条虚构经验手算：本步总奖励 $r_t=0.04$，目标网络给下一观测的三个 Q 值为 $(3,5,4)$，在线网络给本次实际动作的预测为 3。由于这一局尚未真正结束，目标和误差为：

$$
y_t=0.04+0.95\times5=4.79,\qquad e_t=3-4.79=-1.79.
$$

阈值为 1 的 Huber 损失是 $|-1.79|-0.5=1.29$，对预测值的导数为 $-1$；更新方向会推动这次动作的预测值上升。若动作导致死亡，目标直接等于该步记录的总奖励，不再加下一状态价值；总奖励含塑形项，通常不恰好等于基础死亡奖励 $-10$。这只是单样本解释；实际代码把 32 条经验的梯度平均后才更新一次。

Huber 的精确定义及导数见前面的分段公式。`network_gradient` 用 $\operatorname{clip}(e,-1,1)$ 实现这个导数。目标网络只负责算目标数值，`network_gradient` 只对在线网络求导；否则“答案”也会随着同一次反向传播移动。

ε-greedy 在前 50000 步将随机探索率从 1.0 线性降至 0.05。训练仍保留探索；验证、评估和演示的 ε 为零。这是普通 DQN，不是 Double DQN、卷积网络或循环网络。

具体来说，环境步数为 0 时 $\varepsilon=1$，每次都随机抽一个动作；25000 步时约为 $0.525$；到 50000 步后保持 $0.05$。其余时候选在线网络给出最大 Q 值的动作。训练分数可能因探索而波动，不能仅凭某一局变差就认定模型退化。

## 对照代码走完一次训练

下面是 `snake_ai.c` 的主线伪代码；函数名与源码一致，省略了日志、预算检查和错误处理：

```text
初始化 online 网络 → 复制成 target 网络 → 清空 Adam 与 Replay
重复每一局：
    game = snake_reset(新环境种子)
    每一步：
        state = snake_observe(game)
        action = 以 ε 的概率随机动作，否则 greedy_action(online, state)
        outcome = snake_act(game, action)
        next = snake_observe(game)
        replay_push(state, action, outcome.reward, next, outcome.terminated)
        当总步数达到 1000 且为 4 的倍数：
            重复 32 次：从 Replay 随机抽一条经验
                target = td_target(经验, target 网络)
                cache = network_forward(online, 经验.state)
                network_gradient(online, cache, 经验.action, target, 梯度和)
            adam_update(online, 梯度和, 32)
            每完成 500 次参数更新：target 网络 = online 网络
        若死亡、满棋盘或达到本局 4000 步：结束本局
```

对照阅读可以按下表定位，每个函数的“状态变化”决定它属于环境、推理还是训练：

| 步骤 | 源码函数 | 关键输入与结果 |
| --- | --- | --- |
| 创建游戏 | `snake_reset` | 种子 → 初始 `Game` 和食物 |
| 编码局面 | `snake_observe` | `Game` → 11 个二值特征，不修改游戏 |
| 选择动作 | `greedy_action`、`exploration` | 在线网络与随机数 → 0/1/2 |
| 执行并反馈 | `snake_act` | 修改 `Game`，返回奖励、终止和获胜标志 |
| 保存经验 | `replay_push` | 复制 `Transition`，必要时覆盖最旧记录 |
| 建立目标 | `td_target` | 终止时只用奖励，否则使用目标网络的下一状态估值 |
| 学习参数 | `learn_batch` | 抽样、前向、反传、Adam 更新在线网络，周期同步目标网络 |
| 测试策略 | `evaluate` | 固定种子运行多局，只统计分数，不更新参数 |

`train` 每走一步都先存经验，再检查是否该学习。`MAX_STEPS=4000` 表示本局**截断**：这一步经验的 `terminated` 仍为 `false`，学习目标继续估计未来；死亡或填满棋盘才是**真正终止**。总训练预算在局中耗尽时，也不会伪造一条“死亡”的经验。这一区别会直接影响 `td_target` 是否加上未来价值。

## 从零推导训练过程与网络设计

下面按“把游戏写成数学问题 → 设计网络 → 生成学习目标 → 求梯度 → 更新参数 → 实际使用”的顺序推导。公式中的 $g_t$ 表示完整游戏局面，$x_t$ 表示神经网络看到的 11 个数字，$t$ 表示环境步；Adam 公式会另行说明更新步数。

### 1. 把一局贪吃蛇写成数学问题

每一步，环境先处在局面 $g_t$。AI 读取局面得到观测 $x_t$，选择动作 $a_t$；环境执行后产生奖励 $r_t$、新局面 $g_{t+1}$ 和观测 $x_{t+1}$：

$$
g_t \xrightarrow{\text{观察}} x_t
\xrightarrow{\text{选动作 }a_t} g_{t+1},\ r_t,\ x_{t+1}.
$$

本项目的动作集合是 $\mathcal{A}=\{0,1,2\}$，依次表示直行、左转、右转。观测函数 $\phi$ 把完整局面压缩为长度 11 的 0/1 向量：

$$
x_t=\phi(g_t)\in\{0,1\}^{11}.
$$

源码把这次状态转移记录到 `Transition`。下面是 `train` 中的核心片段：

```c
Transition transition = {0};
snake_observe(&game, transition.state);
double epsilon = exploration(total_steps);
transition.action = snake_uniform(&learner->actions_rng) < epsilon
                        ? (int)(snake_random(&learner->actions_rng) % ACTIONS)
                        : greedy_action(&learner->online, transition.state);

SnakeResult outcome = snake_act(&game, transition.action);
transition.reward = outcome.reward;
transition.terminated = outcome.terminated;
snake_observe(&game, transition.next);
replay_push(&learner->replay, &transition);
```

这里 `transition.state` 是 $x_t$，`transition.action` 是 $a_t$，`outcome.reward` 是 $r_t$，`transition.next` 是 $x_{t+1}$。`snake_act` 更新 `game` 后才编码下一观测，所以前后状态的时间顺序与公式一致。实际实现位于 [`snake_ai.c`](../src/games/snake_ai.c) 的 `train`。

这些数字包括三个方向下一格是否碰撞、当前朝向和食物相对方位。它们没有记录蛇身全貌，也不包含精确距离，所以两个不同局面可能产生同一个 $x_t$。完整局面包含预测后续所需的信息；观测是为了让网络输入简单而做的压缩。这种信息缺失是模型可能判断不了长距离路线的原因之一。

`snake_observe` 用布尔比较直接填入数组。以下代码展示危险位、方向位和食物方位位的计算：

```c
for (int action = 0; action < ACTIONS; ++action) {
    state[action] = snake_collision(game, snake_direction(game->direction, action));
}
state[3] = game->direction.x == -1;
state[4] = game->direction.x == 1;
state[5] = game->direction.y == -1;
state[6] = game->direction.y == 1;
state[7] = game->food.x < game->body[0].x;
state[8] = game->food.x > game->body[0].x;
state[9] = game->food.y < game->body[0].y;
state[10] = game->food.y > game->body[0].y;
```

例如 `state[4]` 为 1 表示蛇头朝右；`state[8]` 为 1 表示食物在蛇头右侧。比较表达式的结果会转换为 0 或 1，作为网络的数值输入。完整函数在 [`snake_env.h`](../src/games/snake_env.h) 的 `snake_observe`。

棋盘坐标 $x$ 向右、$y$ 向下。$H$ 是蛇头位置、$F$ 是食物位置；$x_h,y_h$ 和 $x_f,y_f$ 分别是它们的横、纵坐标。两点间曼哈顿距离（只沿横竖方向计格子）为：

$$
d(H,F)=|x_f-x_h|+|y_f-y_h|.
$$

它计算只走水平和竖直格子时的距离，不考虑蛇身障碍。当前方向向量为 $v=(v_x,v_y)$ 时，左、右转分别是 $(v_y,-v_x)$ 和 $(-v_y,v_x)$。例如朝右的方向 $(1,0)$ 左转后是 $(0,-1)$，即屏幕向上。

代入蛇头 $H=(10,5)$ 和食物 $F=(13,3)$：

$$
d(H,F)=|13-10|+|3-5|=|3|+|-2|=3+2=5.
$$

因此前面的 11 项观测虽然能表示“食物在右上”，但无法区分距离是 1 格还是 5 格；势函数会用完整局面另行计算这段距离。

### 2. 设计奖励，让动作有反馈

游戏本身使用基础奖励：吃到食物 $+10$，死亡 $-10$，普通移动 $-0.01$，填满棋盘额外 $+100$。但食物可能很久才吃到；如果只在吃到时奖励，初期大多数动作都得不到反馈。因此代码再加入一个由距离计算的势函数：

$$
\Phi(g)=-\frac{d(H,F)}{20+10}=-\frac{d(H,F)}{30}.
$$

这里 $g$ 是完整游戏局面，$d(H,F)$ 是蛇头到食物的曼哈顿距离，$\Phi(g)$ 是该局面的势；棋盘宽 20、高 10，所以分母为 30。离食物越近，$\Phi$ 越接近零。$r_{\text{base},t}$ 表示第 $t$ 步的原始游戏奖励，不含距离塑形项。单步塑形奖励为：

$$
r_t=r_{\text{base},t}+\gamma\Phi(g_{t+1})-\Phi(g_t),
\qquad \gamma=0.95.
$$

真正死亡或获胜后把下一势设为零；达到 4000 步只是截断，环境仍保留未来价值。举例来说，若一次普通移动让距离从 3 降为 2，则：

$$
r_t=-0.01+0.95\left(-\frac{2}{30}\right)-\left(-\frac{3}{30}\right)
\approx 0.0267.
$$

把中间值逐项展开：旧势为 $-3/30=-0.1$，新势为 $-2/30\approx-0.066667$；新势折扣后是 $0.95\times-0.066667\approx-0.063333$。所以 $-0.01-0.063333+0.1=0.026667$。距离变近带来的约 $0.036667$ 塑形奖励，抵消普通移动的 $-0.01$ 后仍为正数。

对应代码先用 `abs` 算水平和垂直距离，再除以棋盘宽高之和；随后在动作后用新势减旧势：

```c
static inline double snake_potential(const Game *game)
{
    int distance = abs(game->food.x - game->body[0].x)
                 + abs(game->food.y - game->body[0].y);
    return -(double)distance / (WIDTH + HEIGHT);
}

static inline SnakeResult snake_act(Game *game, int action)
{
    double previous = snake_potential(game);
    size_t length = game->length;
    bool moved = step(game, snake_direction(game->direction, action));
    bool won = moved && game->length == CAPACITY;
    bool terminated = !moved || won;
    double reward = !moved ? -10 : (game->length > length ? 10 : -0.01);
    if (won) {
        reward += 100;
    }
    reward += 0.95 * (terminated ? 0 : snake_potential(game)) - previous;
    return (SnakeResult){reward, terminated, won};
}
```

第一段返回 $\Phi(g_t)$；第二段的 `previous` 保存旧势，`snake_potential(game)` 在 `step` 后读取新局面。死亡或获胜时 `terminated` 为真，代码把下一势置零。实际实现还会在获胜时加 100 分，并把结果放入 `SnakeResult`；完整逻辑见 [`snake_act`](../src/games/snake_env.h)。

距离变近后，这一步的总奖励为正。连续 $T$ 步的塑形部分有一个抵消关系：

$$
\sum_{t=0}^{T-1}\gamma^t\left[\gamma\Phi(g_{t+1})-\Phi(g_t)\right]
=-\Phi(g_0)+\gamma^T\Phi(g_T).
$$

中间各步的势会抵消。若终点是真正终止状态且终点势为零，整段回报只多了初始局面相关的 $-\Phi(g_0)$。这是势函数塑形在完整状态假设下的策略不变性直觉；本项目使用部分观测和截断训练，因此不能据此保证学到最优策略。基础奖励、塑形奖励和屏幕显示的食物数是不同概念。

这里 $T$ 是这段轨迹包含的步数，求和下标 $t$ 从 0 到 $T-1$；与梯度公式中表示转置的右上角 $T$ 不同。$g_0$ 是轨迹起点，$g_T$ 是终点局面。

看一个两步的抵消过程。设 $\gamma=0.95$，起点势 $\Phi(g_0)=-0.1$，中间势 $\Phi(g_1)=-2/30\approx-0.066667$，真正终止后的势 $\Phi(g_2)=0$。用 $f_t=\gamma\Phi(g_{t+1})-\Phi(g_t)$ 表示第 $t$ 步的势函数塑形项，则两步分别为：

$$
f_0=0.95(-0.066667)-(-0.1)\approx0.036667,
\qquad
f_1=0.95(0)-(-0.066667)\approx0.066667.
$$

按时间折扣相加，$f_0+0.95f_1\approx0.036667+0.063333=0.1$；右侧也为 $-\Phi(g_0)+0.95^2\Phi(g_2)=0.1$。中间势 $\Phi(g_1)$ 的负项和正项刚好抵消。

### 3. 定义“动作价值”：现在的动作对未来有多好

如果只看眼前一步，AI 可能为了当前食物撞上墙。我们希望它考虑动作带来的整段未来奖励。把未来奖励按时间远近折扣后相加：

$$
G_t=r_t+\gamma r_{t+1}+\gamma^2r_{t+2}+\cdots
=\sum_{k=0}^{\infty}\gamma^k r_{t+k}.
$$

这里 $t$ 是当前环境步，$k$ 是从当前步起向未来数了多少步，$r_{t+k}$ 是第 $t+k$ 步的奖励，$\gamma^k$ 是它的折扣权重。$\gamma=0.95$ 表示 20 步之后的奖励在当前约乘以 $0.95^{20}\approx0.358$；例如 $0.95^{10}\approx0.5987$，再平方得到 $0.95^{20}\approx0.3584$。折扣值越小，越重视近期；越接近 1，越重视较远的结果。

若接下来三步奖励依次为 $1,2,4$，则这三步的折扣回报是：

$$
G_t=1+0.95\times2+0.95^2\times4
=1+1.9+3.61=6.51.
$$

第三步的奖励是 4，但折扣后只贡献 3.61。

给定局面和动作后，未来仍可能受食物随机位置等因素影响，所以用期望表示长期价值：

$$
Q^\pi(g,a)=\mathbb{E}\left[G_t\mid g_t=g,\ a_t=a,\ \pi\right].
$$

这里 $\pi$ 是后续选择动作的策略，$\mathbb{E}$ 是对可能未来结果的加权平均；条件竖线右侧表示只考虑满足给定局面、动作和策略的未来。最优价值 $Q^*(g,a)$ 表示后续采用最佳策略时的价值，星号表示“最优”。贝尔曼方程中的 $a'$ 是下一步候选动作，$g_{t+1}$ 是执行当前动作后的局面。最优策略的关键关系是贝尔曼方程：

$$
Q^*(g,a)=\mathbb{E}\left[
r_t+\gamma\max_{a'\in\mathcal{A}}Q^*(g_{t+1},a')
\mid g_t=g,a_t=a
\right].
$$

它表达“当前动作的价值 = 当前奖励 + 折扣后的下一步最佳价值”。死亡或填满棋盘后没有下一步动作，后续价值取零。我们不知道 $Q^*$，因此接下来用神经网络近似它。

期望是按概率加权：若采取同一个动作后，有一半概率得到回报 4、一半概率得到回报 $-2$，那么：

$$
Q^\pi=\mathbb{E}[G]=0.5\times4+0.5\times(-2)=1.
$$

实际某一局只会得到 4 或 $-2$，1 是许多同类未来的平均价值。再看贝尔曼目标的代入：若当前奖励 $r_t=0.5$、$\gamma=0.5$，下一状态三个动作的 Q 值为 $(2,4,1)$，最佳后续价值为 4，于是：

$$
y_t=0.5+0.5\times\max(2,4,1)=0.5+2=2.5.
$$

若这一步死亡，终止分支直接取 $y_t=r_t$；例如死亡奖励为 $-10$ 时目标就是 $-10$，不会再加 $\gamma\max Q$。

贝尔曼方程中的期望也可以手算：假设下一步有两个等概率结果，最佳后续 Q 值分别为 2 和 8，当前奖励仍为 $0.5$、折扣仍为 $0.5$，则平均后续价值为 $0.5\times2+0.5\times8=5$，期望目标为 $0.5+0.5\times5=3$。训练代码每条经验只观察到其中一个下一状态，所以用该样本目标近似这个期望；经验回放再从许多不同转移中抽样。

源码中的 `td_target` 是这个关系的一步近似：

```c
if (transition->terminated) {
    return transition->reward;
}

Forward next;
network_forward(target, transition->next, &next);
double maximum = next.q[0];
for (int action = 1; action < ACTIONS; ++action) {
    maximum = fmax(maximum, next.q[action]);
}
return transition->reward + 0.95 * maximum;
```

`network_forward` 计算下一观测的三个 Q 值，循环找到最大值；最后一行实现 $r_t+\gamma\max_{a'}Q_{\theta^-}(x_{t+1},a')$。终止经验直接返回奖励，对应终点的后续价值为零。完整代码见 [`snake_ai.c`](../src/games/snake_ai.c) 中的 `td_target`。

### 4. 从输入到三个 Q 值：设计神经网络

本例选择两层各 64 个神经元的全连接网络：$11\to64\to64\to3$。输入 11 个观测，输出三个动作各自的价值估计。网络规模是学习示例中的一个设计选择；数学并不能推出“64 一定最好”，更大的网络也不自动意味着玩得更好。

第 $l$ 层对输入做加权求和并加偏置：

$$
z^{(l)}=W^{(l)}h^{(l-1)}+b^{(l)}.
$$

$W^{(l)}$ 是权重矩阵，$b^{(l)}$ 是偏置向量，$z^{(l)}$ 是激活前的结果，$l$ 是层编号，上标括号表示层而非幂。$\mathbb{R}^{m\times n}$ 表示有 $m$ 行、$n$ 列的实数矩阵。以输入 $(1,2)$、权重 $(0.5,-1)$ 和偏置 $0.3$ 为例，这一行神经元会算出 $z=0.3+0.5\times1-1\times2=-1.2$。矩阵公式只是把同一组乘加对每一行都做一次。隐藏层再通过 ReLU：

$$
\operatorname{ReLU}(z)=\max(0,z),\qquad
h^{(l)}=\operatorname{ReLU}(z^{(l)}).
$$

单个神经元可以直接手算：若两个输入是 $(1,2)$、权重是 $(0.5,-1)$、偏置是 $0.3$，则 $z=0.3+0.5\times1-1\times2=-1.2$，ReLU 输出 $h=\max(0,-1.2)=0$。若另一组参数使 $z=0.6$，则 $h=\max(0,0.6)=0.6$。ReLU 只把负值变为 0，不会把正值压到 1 以内。

令 $h^{(0)}=x$，本网络的完整计算为：

$$
\begin{aligned}
z^{(1)}&=W^{(1)}x+b^{(1)}, & h^{(1)}&=\operatorname{ReLU}(z^{(1)}),\\
z^{(2)}&=W^{(2)}h^{(1)}+b^{(2)}, & h^{(2)}&=\operatorname{ReLU}(z^{(2)}),\\
q&=W^{(3)}h^{(2)}+b^{(3)}.
\end{aligned}
$$

`dense` 把矩阵的一行展开为内层循环的乘加；`network_forward` 再按层调用它：

```c
for (int row = 0; row < outputs; ++row) {
    double value = bias[row];
    for (int column = 0; column < inputs; ++column) {
        value += weights[row * inputs + column] * input[column];
    }
    result[row] = relu ? fmax(0, value) : value;
}

dense(net->p + W1, net->p + B1, state, FEATURES, HIDDEN, cache->h1, true);
dense(net->p + W2, net->p + B2, cache->h1, HIDDEN, HIDDEN, cache->h2, true);
dense(net->p + W3, net->p + B3, cache->h2, HIDDEN, ACTIONS, cache->q, false);
```

`weights[row * inputs + column]` 是把矩阵按行连续存储时的下标。前两次调用末尾为 `true`，表示应用 ReLU；输出层为 `false`，保留正负 Q 值。完整函数在 [`snake_ai_network.h`](../src/games/snake_ai_network.h) 的 `dense` 和 `network_forward`。

矩阵形状分别为 $W^{(1)}\in\mathbb{R}^{64\times11}$、$W^{(2)}\in\mathbb{R}^{64\times64}$、$W^{(3)}\in\mathbb{R}^{3\times64}$；输出 $q\in\mathbb{R}^{3}$。$q_a$ 是动作 $a$ 的 Q 值。输出层不接 ReLU，允许 Q 值为负；三个数也不是概率，不要求加起来等于 1。

参数总数可以直接按矩阵元素和偏置数相加：

$$
(64\times11+64)+(64\times64+64)+(3\times64+3)=5123.
$$

逐层数一下：第一层有 $64\times11=704$ 个权重和 64 个偏置，共 768；第二层有 $64\times64=4096$ 个权重和 64 个偏置，共 4160；输出层有 $3\times64=192$ 个权重和 3 个偏置，共 195。相加 $768+4160+195=5123$ 个可训练参数。

`Network.p` 是将这些矩阵和偏置依次铺平的一维数组。网络前向计算完成后，贪心动作是 $\operatorname{argmax}_{a\in\mathcal{A}}q_a$，即 Q 最大的动作编号。

完整三层结构也可以用每层各一个隐藏神经元的小网络演算：输入 $x=1$，第一层 $W^{(1)}=2,b^{(1)}=-1$，所以 $z^{(1)}=2\times1-1=1$、$h^{(1)}=1$；第二层 $W^{(2)}=3,b^{(2)}=-1$，所以 $z^{(2)}=3\times1-1=2$、$h^{(2)}=2$。令输出权重为 $(0.5,-1,2)$、偏置为 $(0,1,-1)$，则：

$$
q=(0.5\times2+0,\;-1\times2+1,\;2\times2-1)=(1,-1,3),
\qquad \operatorname{argmax}_a q_a=2.
$$

这和生产网络的计算顺序完全相同，只是每层神经元更少；生产网络前向传播的完整数值过程见后面的玩具网络算例。

#### 为什么权重需要随机初始化

如果同一层所有神经元的权重完全相同，它们会算出相同结果，得到相同梯度，训练中仍保持相同，网络就浪费了神经元。代码用近似标准正态随机数初始化权重。Box–Muller 变换把两个独立均匀随机数 $u_1\in(0,1)$、$u_2\in[0,1)$ 转为近似标准正态样本 $\xi$：

$$
\xi=\sqrt{-2\ln u_1}\cos(2\pi u_2),\qquad
\xi\sim\mathcal{N}(0,1).
$$

这里 $u_1,u_2$ 是均匀随机输入，$\xi$ 是生成的标准正态样本，$\mathcal{N}(0,1)$ 表示均值为 0、方差为 1 的正态分布；此处的 $\pi$ 是圆周率，和策略符号 $\pi$ 无关。它特意使用 $\xi$，避免与网络中表示“激活前结果”的 $z^{(l)}$ 混淆。

`gaussian` 先让 $u_1$ 严格大于零，再按 Box–Muller 公式计算；$6.283185307179586$ 是 $2\pi$ 的近似值：

```c
double u = ((double)snake_random(rng) + 1) / 4294967297.0;
return sqrt(-2 * log(u)) * cos(6.283185307179586 * snake_uniform(rng));
```

这里 `u` 对应 $u_1$，`snake_uniform(rng)` 对应 $u_2$；源码使用 `log`、`sqrt`、`cos` 三个 C 数学库函数实现公式。

取易算的 $u_1=0.5,u_2=0$，则 $\cos(0)=1$，所以：

$$
\xi=\sqrt{-2\ln(0.5)}\times1
=\sqrt{1.386294}\approx1.17741.
$$

所以这次算出的标准正态样本为 $\xi\approx1.17741$。真实程序从伪随机数发生器取 $u_1,u_2$，不会固定使用这组数。

标准正态数均值为零、方差为一。第一、二层使用 He 缩放；若一层有 $n$ 个输入，则每个权重按下式生成：

$$
W_{ij}\sim\mathcal{N}\left(0,\frac{2}{n}\right)
\quad\Longleftrightarrow\quad
W_{ij}=\xi\sqrt{\frac{2}{n}}.
$$

这里 $W_{ij}$ 是第 $i$ 个输出神经元连接第 $j$ 个输入的权重，$n$ 是该层输入数，$\xi$ 是前面生成的标准正态样本。记号 $\mathcal{N}(\mu,\sigma^2)$ 表示均值为 $\mu$、方差为 $\sigma^2$ 的正态分布，所以此处权重均值为 0、方差为 $2/n$。为什么方差取 $2/n$？假设输入互相独立、均值为零、方差为 $v_x$，权重互相独立、均值为零、方差为 $\sigma_w^2$，则加权和的方差近似为 $n\sigma_w^2v_x$。对称输入经过 ReLU 后，约一半信号被置零，输出的平方均值约减半。希望下一层仍保持约 $v_x$ 的信号量级，就令 $n\sigma_w^2v_x/2\approx v_x$，得到 $\sigma_w^2\approx2/n$。这就是 He 缩放的直觉推导。因此两层隐藏层分别使用 $n=11$ 和 $n=64$。线性输出层使用较小的方差 $1/64$，偏置全部设为零。初始化只提供合理起点，不保证训练收敛。这里 $v_x$ 表示输入信号的方差，不是方向向量或 Adam 状态中的 $v$。

源码把标准正态样本 $\xi$ 乘以对应标准差 $\sqrt{2/n}$；因为“标准差的平方 = 方差”，所以得到上面的 $2/n$ 方差。以第一层为例，$n=11$，权重方差为 $2/11\approx0.181818$，标准差为 $\sqrt{2/11}\approx0.426401$。若刚才的标准正态样本 $\xi\approx1.17741$ 被抽到，这个权重约为 $1.17741\times0.426401=0.5020$。第二隐藏层有 $n=64$，方差为 $2/64=0.03125$、标准差约为 $0.176777$；线性输出层权重方差为 $1/64=0.015625$、标准差为 $1/8=0.125$。作第一层方差推导的直观检查：11 个独立输入的方差都约为 1 时，加权和方差约为 $11\times(2/11)\times1=2$；ReLU 约去掉一半平方信号后约为 1，回到原输入的量级。此处是解释初始化尺度的近似，不代表每一次随机初始化都会精确得到这些方差：

```c
for (int i = W1; i < B1; ++i) {
    net->p[i] = gaussian(rng) * sqrt(2.0 / FEATURES);
}
for (int i = W2; i < B2; ++i) {
    net->p[i] = gaussian(rng) * sqrt(2.0 / HIDDEN);
}
for (int i = W3; i < B3; ++i) {
    net->p[i] = gaussian(rng) * sqrt(1.0 / HIDDEN);
}
```

`FEATURES` 是 11，`HIDDEN` 是 64。第三层使用 $\sqrt{1/64}$，对应输出权重方差 $1/64$。这里的 `gaussian` 通过 Box–Muller 公式生成伪随机样本，代码见同一头文件的 `gaussian`。

### 5. 用目标网络构造一个可学习的答案

对每条经验，在线网络给当前动作预测 $Q_\theta(x_t,a_t)$。下一状态价值由目标网络 $Q_{\theta^-}$ 计算，训练目标为：

$$
y_t=
\begin{cases}
r_t, & \text{若经验标记为真正终止},\\
r_t+\gamma\displaystyle\max_{a'\in\mathcal{A}}
Q_{\theta^-}(x_{t+1},a'), & \text{否则}.
\end{cases}
$$

目标网络参数 $\theta^-$ 每 500 次参数更新才从在线网络 $\theta$ 复制一次。它让训练目标在这段时间内相对稳定。这里用目标网络估值但仍直接取其最大值；Double DQN 会把“选哪个动作”和“评估该动作”分给两张网络，本项目没有实现 Double DQN。

在 `learn_batch` 中，目标网络只提供 `td_target` 的数值，随后在线网络的 `network_gradient` 用该目标计算梯度；达到 500 次更新时才复制参数：

```c
total += network_gradient(&learner->online, &cache, transition->action,
                          td_target(transition, &learner->target), &gradient);

if (learner->adam.updates % 500 == 0) {
    learner->target = learner->online;
}
```

这两行对应“旧目标网络算 $y_t$，在线网络按 $Q_\theta-y_t$ 更新；周期性令 $\theta^-\leftarrow\theta$”。结构体赋值会复制整组 5123 个参数。

经验池保存 $ (x_t,a_t,r_t,x_{t+1},\text{terminated}) $，容量为 10000。前 1000 个环境步只收集经验；之后每 4 步从池中有放回抽 32 条。抽样有放回意味着同一条经验可以在同一批重复出现。回放打散相邻时间步的强相关，但不会让经验真正独立，也不保证状态分布均匀。

`learn_batch` 的循环里用一个随机槽位索引取出经验，直到组成 32 条批次：

```c
for (int i = 0; i < BATCH; ++i) {
    size_t index = snake_random(&learner->sample_rng) % learner->replay.count;
    const Transition *transition = &learner->replay.data[index];
    Forward cache;
    network_forward(&learner->online, transition->state, &cache);
    total += network_gradient(&learner->online, &cache, transition->action,
                              td_target(transition, &learner->target), &gradient);
}
```

`% replay.count` 将随机整数映射到经验池现有槽位；每次抽样后没有移除该槽位，所以是有放回采样。`gradient` 在循环前清零，每次 `network_gradient` 用 `+=` 累积新的偏导。

注意 `terminated` 只表示死亡或获胜。4000 步上限属于截断，仍使用公式第二行继续估值；如果把截断误标成终止，会错误地把未来价值删除。代码对每条经验只抽到一个下一状态样本，用它近似贝尔曼方程中的期望。

### 6. 用损失衡量预测差距

让在线网络预测已执行动作的价值，并与目标比较：

$$
e_t=Q_\theta(x_t,a_t)-y_t.
$$

这里 $Q_\theta(x_t,a_t)$ 是在线网络对实际执行动作的预测，$y_t$ 是目标网络产生的学习目标；两者之差 $e_t$ 是带符号的预测误差，正值表示预测偏高，负值表示预测偏低。

平方误差会让很大的误差产生非常大的梯度。代码改用阈值为 1 的 Huber 损失：

$$
L(e)=
\begin{cases}
\frac{1}{2}e^2, & |e|\leq1,\\
|e|-\frac{1}{2}, & |e|>1.
\end{cases}
$$

这里 $L(e)$ 是以误差 $e$ 为输入的损失函数，$|e|$ 是误差大小，阈值 1 决定使用哪一段曲线。对误差求导：

$$
\frac{\partial L}{\partial e}=
\begin{cases}
e, & |e|\leq1,\\
\operatorname{sign}(e), & |e|>1,
\end{cases}
\quad=\quad \operatorname{clip}(e,-1,1).
$$

这里 $\operatorname{sign}(e)$ 在 $e$ 为正、零、负时分别取 $1,0,-1$；$\operatorname{clip}(e,-1,1)$ 把小于 $-1$ 的值改为 $-1$、大于 1 的值改为 1，其余保持原值。因此它就是 Huber 损失对误差的导数。

源码把这两个公式写成一个损失函数和一行裁剪：

```c
static inline double huber(double error)
{
    return fabs(error) <= 1 ? 0.5 * error * error : fabs(error) - 0.5;
}

double error = cache->q[action] - target;
double derivative = fmax(-1, fmin(1, error));
```

例如误差为 $-1.79$，`huber` 返回 $1.29$，而嵌套的 `fmin`/`fmax` 把导数限制为 $-1$。误差为 $-0.5$ 时，损失是 $0.125$，导数仍是 $-0.5$。代码见 [`huber` 和 `network_gradient`](../src/games/snake_ai_network.h)。

小误差仍用平滑的平方曲线，大误差的导数限制在 $-1$ 到 $1$ 之间。目标 $y_t$ 在求导时视为常数；梯度只回到在线网络，不能让目标网络通过同一条梯度移动。

### 7. 从输出误差逐层反传到权重

设输出层的三个 Q 值为 $q_0,q_1,q_2$，本条经验执行的动作是 $a$。本例只训练 $q_a$：

下面 $k$ 在输出层表示动作编号；$i$ 表示第二隐藏层神经元，$j$ 表示第一隐藏层神经元；在第一层权重 $W^{(1)}_{jk}$ 中，$k$ 改为输入特征编号。$d$ 是 Huber 损失对所选动作 Q 值的导数。

$$
d=\frac{\partial L}{\partial q_a}
=\operatorname{clip}(e,-1,1),\qquad
\delta^{(3)}_k=
\begin{cases}
d, & k=a,\\
0, & k\neq a.
\end{cases}
$$

输出层 $q=W^{(3)}h^{(2)}+b^{(3)}$，所以每个输出权重的导数是“输出误差乘它接收的输入”：

$$
\frac{\partial L}{\partial b^{(3)}_k}=\delta^{(3)}_k,
\qquad
\frac{\partial L}{\partial W^{(3)}_{ki}}
=\delta^{(3)}_k h^{(2)}_i.
$$

再把误差传回第二隐藏层。ReLU 正数处导数为 1，负数处导数为 0；代码在零处也取 0。用 $\mathbf{1}[z>0]$ 表示条件成立为 1、否则为 0：

$$
\delta^{(2)}_i
=\mathbf{1}[z^{(2)}_i>0]\,W^{(3)}_{ai}d,
\qquad
\frac{\partial L}{\partial b^{(2)}_i}=\delta^{(2)}_i,
\qquad
\frac{\partial L}{\partial W^{(2)}_{ij}}
=\delta^{(2)}_i h^{(1)}_j.
$$

同样地，把第二层传来的影响乘以第二层权重，再经过第一层 ReLU：

$$
\delta^{(1)}_j
=\mathbf{1}[z^{(1)}_j>0]
\sum_{i=1}^{64}W^{(2)}_{ij}\delta^{(2)}_i,
\qquad
\frac{\partial L}{\partial b^{(1)}_j}=\delta^{(1)}_j,
\qquad
\frac{\partial L}{\partial W^{(1)}_{jk}}
=\delta^{(1)}_j x_k.
$$

把三层都缩成每层一个神经元，手算一次这组反传公式。令输入 $x=2$；第一层 $W^{(1)}=1.5,b^{(1)}=0$，所以 $z^{(1)}=3,h^{(1)}=3$；第二层 $W^{(2)}=2,b^{(2)}=0$，所以 $z^{(2)}=6,h^{(2)}=6$。动作 0 的输出权重为 $W^{(3)}_0=0.5$、偏置为 0，因此 $q_0=0.5\times6=3$。若目标 $y=4$，误差 $e=3-4=-1$，Huber 导数 $d=-1$。两个 ReLU 都处在正数区间，导数都是 1，于是：

$$
\begin{aligned}
\delta^{(3)}_0&=-1, &
\frac{\partial L}{\partial b^{(3)}_0}&=-1, &
\frac{\partial L}{\partial W^{(3)}_0}&=(-1)\times6=-6,\\
\delta^{(2)}&=1\times0.5\times(-1)=-0.5, &
\frac{\partial L}{\partial b^{(2)}}&=-0.5, &
\frac{\partial L}{\partial W^{(2)}}&=(-0.5)\times3=-1.5,\\
\delta^{(1)}&=1\times2\times(-0.5)=-1, &
\frac{\partial L}{\partial b^{(1)}}&=-1, &
\frac{\partial L}{\partial W^{(1)}}&=(-1)\times2=-2.
\end{aligned}
$$

每往前一层，就把当前误差乘该层权重，再乘 ReLU 的导数；权重梯度还要乘该权重的输入。若某个预激活值是负数，对应 ReLU 导数为 0，该神经元传回前层的梯度也会变成 0。

`network_gradient` 按相同顺序计算这些值。以下摘录省略函数签名和损失返回部分，索引偏移 `W3`、`B3` 等分别指向平坦参数数组中的矩阵和偏置段：

```c
gradient->p[B3 + action] += derivative;
double delta2[HIDDEN];
for (int i = 0; i < HIDDEN; ++i) {
    gradient->p[W3 + action * HIDDEN + i] += derivative * cache->h2[i];
    delta2[i] = cache->h2[i] > 0 ? derivative * net->p[W3 + action * HIDDEN + i] : 0;
    gradient->p[B2 + i] += delta2[i];
    for (int j = 0; j < HIDDEN; ++j) {
        gradient->p[W2 + i * HIDDEN + j] += delta2[i] * cache->h1[j];
    }
}
```

第一行是 $\partial L/\partial b^{(3)}_a=d$；第二行把输出误差乘隐藏层输入，得到 $\partial L/\partial W^{(3)}_{ai}$。`delta2[i]` 是 $\delta^{(2)}_i$，条件 `cache->h2[i] > 0` 正是 ReLU 导数。内层循环把第二隐藏层误差乘 `h1[j]`，累积到 $W^{(2)}_{ij}$。第一层的循环也按同样模式计算，只是再多一次把误差乘回 $W^{(2)}$。

第一隐藏层的反传在源码中进一步累加从 64 个第二层神经元传来的影响：

```c
for (int j = 0; j < HIDDEN; ++j) {
    double delta1 = 0;
    for (int i = 0; i < HIDDEN; ++i) {
        delta1 += delta2[i] * net->p[W2 + i * HIDDEN + j];
    }
    if (cache->h1[j] <= 0) {
        delta1 = 0;
    }
    gradient->p[B1 + j] += delta1;
    for (int i = 0; i < FEATURES; ++i) {
        gradient->p[W1 + j * FEATURES + i] += delta1 * cache->input[i];
    }
}
```

内层求和对应公式中的 $\sum_i W^{(2)}_{ij}\delta^{(2)}_i$；`cache->h1[j] <= 0` 时清零，正是第一层 ReLU 的导数。最后一层循环把 $\delta^{(1)}_j x_k$ 写入一维 `gradient->p`，这里 `i` 是输入下标 $k$。

这就是链式法则在三层网络中的具体形式：每一层先汇总后续误差，再乘本层激活函数的导数；权重的导数还要乘该权重的输入。其他两个动作没有直接的输出层误差，但由于它们共享隐藏层，隐藏权重变化后，其 Q 值也可能改变。`network_gradient` 将这些偏导累加到梯度数组，不在此函数里更新权重。

测试会用有限差分检查反向传播。对参数 $\theta_i$ 加减很小的 $h$，用损失变化估算导数：

$$
\frac{\partial L}{\partial\theta_i}
\approx\frac{L(\theta_i+h)-L(\theta_i-h)}{2h},
\qquad h=10^{-6}.
$$

这里 $\theta_i$ 是第 $i$ 个网络参数，$h$ 是一个很小的正扰动量；计算 $L(\theta_i\pm h)$ 时其余参数保持不变。分子用参数分别增加和减少 $h$ 时的损失差，分母 $2h$ 把差值换算成单位参数变化的斜率。这个 $h$ 是差分步长，不是隐藏层激活 $h^{(l)}$。

测试将该估算与解析梯度比较；正常训练用上面的反向传播，不会逐个参数重新运行网络。以 $L(\theta)=\theta^2$、$\theta=3$、$h=0.001$ 为例：

$$
\frac{L(3.001)-L(2.999)}{0.002}
=\frac{9.006001-8.994001}{0.002}
=\frac{0.012}{0.002}=6.
$$

解析导数也是 $2\theta=6$。实际测试使用更小的 $h=10^{-6}$ 和浮点数，存在舍入误差，所以比较时使用容差，而不要求两者逐位相等。

### 8. 对一批经验求平均并用 Adam 更新

一个批次包含 $B=32$ 条经验。这里 $B$ 是批次样本数，$j$ 标记第几条经验，$L_j$ 是该经验的损失；$i$ 标记网络参数位置，$\partial L_j/\partial\theta_i$ 是这条经验对第 $i$ 个参数产生的梯度。批量平均损失和参数梯度为：

$$
L_{\text{batch}}=\frac{1}{B}\sum_{j=1}^{B}L_j,
\qquad
g_i=\frac{1}{B}\sum_{j=1}^{B}
\frac{\partial L_j}{\partial\theta_i}.
$$

因此 $L_{\text{batch}}$ 是一个批次的平均损失，$g_i$ 是第 $i$ 个参数在这个批次上的平均梯度。

例如一个只有 3 条样本的小批次，损失是 $0.2,0.4,0.6$，平均损失为 $(0.2+0.4+0.6)/3=0.4$。某一个参数的三条样本梯度若是 $-2,1,4$，则这个参数送给优化器的平均梯度为 $(-2+1+4)/3=1$。项目批次大小为 32，计算方式相同，只是求和项更多。

在 C 中，`learn_batch` 把 32 条样本的损失相加；梯度先累加在 `gradient` 中，再把批次大小传给 `adam_update`：

```c
*loss = total / BATCH;
if (!isfinite(*loss) || !adam_update(&learner->online, &learner->adam,
                                     &gradient, BATCH)) {
    return false;
}
```

`total / BATCH` 对应 $L_{\text{batch}}$；梯度累加器中的每个分量也在 `adam_update` 中除以 `batch`，对应 $g_i$。

先把 $g_i$ 从第 1 个参数到第 5123 个参数排成向量 $g$，再计算其整体长度（欧几里得范数）：

$$
\lVert g\rVert_2=\sqrt{\sum_{i=1}^{5123}g_i^2}.
$$

若它大于 10，所有分量统一乘以 $10/\lVert g\rVert_2$；否则保持不变：

$$
\tilde g_i=
\begin{cases}
g_i, & \lVert g\rVert_2\leq10,\\
g_i\dfrac{10}{\lVert g\rVert_2}, & \lVert g\rVert_2>10.
\end{cases}
$$

代码先把梯度和除以批次大小，再累计平方和；比较平方范数与 $10^2=100$，避免为了比较而先开平方：

```c
for (int i = 0; i < PARAMETERS; ++i) {
    gradient->p[i] /= (double)batch;
    norm_squared += gradient->p[i] * gradient->p[i];
}
double scale = norm_squared > 100 ? 10 / sqrt(norm_squared) : 1;
```

当平方范数大于 100，`scale` 就是 $10/\lVert g\rVert_2$；否则为 1。随后乘到每个分量上，等价于分段公式。

这是整体梯度裁剪，方向不变，整体长度至多为 10。例如只有两个梯度分量 $g=(9,12)$，则 $\lVert g\rVert_2=\sqrt{9^2+12^2}=15$，超过上限 10；缩放因子是 $10/15=2/3$，裁剪后 $\tilde g=(6,8)$，长度为 $\sqrt{6^2+8^2}=10$。两个分量按同一比例缩小，所以方向不变。前面的完整训练算例则是 $\lVert g\rVert_2\approx0.342$，没有触发裁剪。之后 Adam 根据过去的梯度，为每个参数估计平均方向和平方梯度。下面 $i$ 是参数下标，$t$ 是参数更新次数，$\tilde g_{i,t}$ 是该次更新用的裁剪后梯度，$m_{i,t}$ 和 $v_{i,t}$ 分别是梯度及梯度平方的指数移动平均；初值为 $m_{i,0}=v_{i,0}=0$，$\beta_1=0.9$、$\beta_2=0.999$ 控制平均值对新梯度的权重：

$$
\begin{aligned}
m_{i,t}&=0.9m_{i,t-1}+0.1\tilde g_{i,t},\\
v_{i,t}&=0.999v_{i,t-1}+0.001\tilde g_{i,t}^{\,2},\\
\widehat m_{i,t}&=\frac{m_{i,t}}{1-0.9^t},\qquad
\widehat v_{i,t}=\frac{v_{i,t}}{1-0.999^t},\\
\theta_{i,t}&=\theta_{i,t-1}
-0.001\frac{\widehat m_{i,t}}
{\sqrt{\widehat v_{i,t}}+10^{-8}}.
\end{aligned}
$$

$\widehat m_{i,t}$ 和 $\widehat v_{i,t}$ 是偏差校正后的一阶、二阶平均；$\theta_{i,t-1}$ 是更新前参数，$\theta_{i,t}$ 是更新后参数。公式中的 $0.001$ 就是学习率 $\eta$，这里只是直接写出项目采用的数值。

更新语句与公式逐项对应：

```c
adam->m[i] = 0.9 * adam->m[i] + 0.1 * g;
adam->v[i] = 0.999 * adam->v[i] + 0.001 * g * g;
double m = adam->m[i] / (1 - adam->beta1_power);
double v = adam->v[i] / (1 - adam->beta2_power);
net->p[i] -= 0.001 * m / (sqrt(v) + 1e-8);
```

`g` 是已平均并裁剪的当前梯度，`beta1_power` 和 `beta2_power` 分别保存 $0.9^t$ 与 $0.999^t$。代码把校正后的值放入局部变量 `m`、`v`，再更新对应的 `net->p[i]` 参数。完整函数为 [`adam_update`](../src/games/snake_ai_network.h)。

$m$ 是梯度的指数移动平均，$v$ 是梯度平方的指数移动平均；它们分别保留近期方向和近期幅度信息，$1-\beta^t$ 修正这两个平均从零开始时的早期偏小，$10^{-8}$ 防止除以零，$0.001$ 是学习率。第一次更新若 $g=1$，则 $m=0.1$、$v=0.001$；因为 $t=1$，校正后 $\widehat m=0.1/(1-0.9)=1$、$\widehat v=0.001/(1-0.999)=1$。所以参数变化约为 $-0.001\times1/(1+10^{-8})\approx-0.001$。若 $g=-0.112$，则校正后分别为 $-0.112$ 和 $0.112^2$，参数约增加 $0.001$，与后面完整训练算例一致。源码会检查数值是否有限；如果异常发生在更新中途，当前实现不承诺回滚已更新参数。

### 9. 把一条经验从奖励算到梯度

前面的公式涉及 5123 个参数，直接拿真实网络手算会很困难。先把网络缩小为“1 个输入 → 1 个 ReLU 神经元 → 3 个动作输出”。它保留了项目中的全连接、ReLU、线性 Q 输出和反向传播规则，但减少了维度，也少一层隐藏层。反向传播会在每层重复同一种链式计算，因此这个小网络足以看清每一步；真实网络会再多经过一个 64 神经元隐藏层。

假设当前观测只有一个数 $x=1$。玩具网络参数取如下值。为贴近生产代码，输出层仍标为第 3 层 $W^{(3)},b^{(3)}$；玩具网络省略了生产网络的第二隐藏层，因此它的输出层标号中间跳过 2。

$$
W^{(1)}=0.5,\quad b^{(1)}=0.1,\qquad
W^{(3)}=(1,-0.5,0.2),\quad
b^{(3)}=(0,0.1,-0.1).
$$

#### 第一步：前向计算三个动作价值

隐藏神经元先做乘加，再通过 ReLU：

$$
z^{(1)}=W^{(1)}x+b^{(1)}=0.5\times1+0.1=0.6,
\qquad h^{(1)}=\max(0,0.6)=0.6.
$$

接着每个输出各自乘隐藏值并加偏置：

$$
\begin{aligned}
Q(x,0)&=1\times0.6+0=0.6,\\
Q(x,1)&=(-0.5)\times0.6+0.1=-0.2,\\
Q(x,2)&=0.2\times0.6-0.1=0.02.
\end{aligned}
$$

因此贪心动作是 $a_t=0$，也就是直行。生产代码用 `dense` 的循环算相同乘加，再由 `greedy_action` 找最大 Q；这里把矩阵循环展开，直接写出每个数：

```c
double z1 = 0.5 * 1.0 + 0.1;
double h1 = fmax(0.0, z1);
double q[3] = {
    1.0 * h1 + 0.0,
    -0.5 * h1 + 0.1,
    0.2 * h1 - 0.1
};
```

#### 第二步：执行动作并算出奖励

假设这次直行让蛇头到食物的距离由 3 格变为 2 格，没有吃到食物，也没有碰撞。旧势和新势为：

$$
\Phi(g_t)=-\frac{3}{30}=-0.1,
\qquad
\Phi(g_{t+1})=-\frac{2}{30}\approx-0.066667.
$$

普通移动基础奖励为 $-0.01$，所以总奖励为：

$$
r_t=-0.01+0.95\times(-0.066667)-(-0.1)
\approx0.026667.
$$

这就是 [`snake_act`](../src/games/snake_env.h) 返回的 `outcome.reward`。代码先存 `previous`，移动后重新计算势，并用终止标志决定下一势是否置零；因此本例没有终止时使用新势。

#### 第三步：用目标网络算学习目标

假设目标网络对下一观测算出三个 Q 值 $(0.8,0.3,0.1)$。最大值是 $0.8$，这条经验未终止，因此：

$$
y_t=r_t+\gamma\max_{a'}Q_{\theta^-}(x_{t+1},a')
=0.026667+0.95\times0.8
\approx0.786667.
$$

生产代码先由 `network_forward(target, ...)` 算出 `next.q`，再循环找最大值，最后执行 `transition->reward + 0.95 * maximum`。对应实现是 [`td_target`](../src/games/snake_ai.c)。如果经验是真正终止，代码直接令 $y_t=r_t$，不加这项未来估值。

例如同一旧局面的势为 $-0.1$，若这一步撞墙死亡，则基础奖励是 $-10$、终止后的势按规则为 0：

$$
r_t=-10+0-(-0.1)=-9.9,
\qquad y_t=r_t=-9.9.
$$

这个分支不使用目标网络的 Q 值；这也说明终止标志会改变学习目标的计算方式。

#### 第四步：算误差与 Huber 损失

在线网络对已执行动作 0 的预测是 $Q_\theta(x_t,0)=0.6$。误差为：

$$
e_t=Q_\theta(x_t,a_t)-y_t
=0.6-0.786667=-0.186667.
$$

因为 $|e_t|<1$，使用 Huber 的平方分支：

$$
L(e_t)=\frac{1}{2}e_t^2
=\frac{1}{2}(-0.186667)^2
\approx0.017422,
\qquad
d=\frac{\partial L}{\partial Q_\theta(x_t,0)}=-0.186667.
$$

在 [`network_gradient`](../src/games/snake_ai_network.h) 中，前两句具体对应：

```c
double error = cache->q[action] - target;
double derivative = fmax(-1, fmin(1, error));
```

`error` 是 $e_t$；这里误差小于 1，`derivative` 就等于误差本身。目标 `target` 在反向传播中固定不变。

#### 第五步：把误差传回网络参数

本例中输出动作 0 的权重是 1，隐藏值是 0.6。输出偏置梯度、输出权重梯度分别为：

$$
\frac{\partial L}{\partial b^{(3)}_0}=d=-0.186667,
\qquad
\frac{\partial L}{\partial W^{(3)}_0}=d h^{(1)}
=-0.186667\times0.6=-0.112.
$$

隐藏神经元的预激活值 $z^{(1)}=0.6>0$，所以 ReLU 导数为 1。误差传回隐藏层后：

$$
\delta^{(1)}=dW^{(3)}_0\times1
=-0.186667\times1=-0.186667.
$$

再乘隐藏层输入 $x=1$，得到第一层权重和偏置梯度：

$$
\frac{\partial L}{\partial W^{(1)}}=\delta^{(1)}x=-0.186667,
\qquad
\frac{\partial L}{\partial b^{(1)}}=\delta^{(1)}=-0.186667.
$$

因此本例只有四个非零梯度：

$$
\nabla_\theta L=
(-0.186667,\,-0.186667,\,-0.112,\,-0.186667),
$$

依次对应 $W^{(1)},b^{(1)},W^{(3)}_0,b^{(3)}_0$。另外两个动作的输出权重和偏置没有直接梯度；真实网络中的其它输出同样如此，但共享的隐藏层参数会影响三个动作的 Q 值。`network_gradient` 将这些值加到 `gradient->p`，尚未立即修改网络。

#### 第六步：裁剪并进行第一次 Adam 更新

为了让算术简单，假设批次只有这一条样本，即 $B=1$。真实项目每批使用 $B=32$ 条经验，会先求平均。当前梯度的整体长度为：

$$
\lVert g\rVert_2
=\sqrt{3\times(-0.186667)^2+(-0.112)^2}
\approx0.342<10.
$$

所以不触发范数裁剪。Adam 第一次更新时，偏差校正后 $\widehat m=g$、$\widehat v=g^2$。对任意一个负梯度，例如 $g=-0.112$：

$$
m=0.1g=-0.0112,\qquad
v=0.001g^2=0.000012544,
\qquad
\widehat m=-0.112,\quad\widehat v=0.112^2.
$$

代入学习率 $\eta=0.001$ 的更新式：

$$
\theta_{\text{new}}
=\theta_{\text{old}}
-0.001\frac{-0.112}{\sqrt{0.112^2}+10^{-8}}
\approx\theta_{\text{old}}+0.001.
$$

因此四个负梯度对应的参数在这一步都约增加 $0.001$：

$$
W^{(1)}:0.5\to0.501,\quad
b^{(1)}:0.1\to0.101,\quad
W^{(3)}_0:1\to1.001,\quad
b^{(3)}_0:0\to0.001.
$$

这是 Adam 第一次更新的近似结果；$10^{-8}$ 会让增量比 $0.001$ 略小。生产代码按每个参数分别计算一阶、二阶平均，并在 [`adam_update`](../src/games/snake_ai_network.h) 中更新后增加 `adam.updates`。

#### 第七步：重新前向，模型如何使用参数

用更新后的第一层参数重算：

$$
z^{(1)}_{\text{new}}=0.501\times1+0.101=0.602,
\qquad h^{(1)}_{\text{new}}=0.602.
$$

三个动作的新 Q 值约为：

$$
\begin{aligned}
Q_{\text{new}}(x,0)&=1.001\times0.602+0.001=0.603602,\\
Q_{\text{new}}(x,1)&=-0.5\times0.602+0.1=-0.201,\\
Q_{\text{new}}(x,2)&=0.2\times0.602-0.1=0.0204.
\end{aligned}
$$

所以模型仍会选动作 0。训练时之后还会重复“收集经验 → 算目标 → 计算损失与梯度 → 更新参数”；`play` 加载模型后只执行前向传播和 [`greedy_action`](../src/games/snake_ai_network.h) 的最大值选择，不再做反向传播或 Adam 更新。

这个例子展示了公式之间的完整因果关系：游戏奖励构造目标，目标与网络预测构成误差，误差经链式法则变成梯度，Adam 再改动参数。真实项目把单个隐藏神经元扩展到两层各 64 个神经元，把一个样本扩展到 32 条批次；每一步仍按相同的乘加、比较、求和和更新规则执行。

### 10. 探索训练与无学习推理

只按当前最大 Q 选动作，网络可能很早就固定在一个糟糕路线。因此训练阶段采用 ε-greedy：以概率 $\varepsilon$ 随机选动作，以概率 $1-\varepsilon$ 选最大 Q 的动作。本项目用 $n$ 表示已经执行的环境步数，并在第 $n$ 步使用：

$$
\varepsilon(n)=1-0.95\min\left(1,\frac{n}{50000}\right).
$$

对应的 `exploration` 函数把步数先除以 50000，再用 `fmin` 限制最大值为 1：

```c
static double exploration(size_t steps)
{
    return 1 - 0.95 * fmin(1, (double)steps / 50000);
}
```

训练循环再用一个均匀随机数和 $\varepsilon$ 比较，决定探索还是贪心选择：

```c
transition.action = snake_uniform(&learner->actions_rng) < epsilon
                        ? (int)(snake_random(&learner->actions_rng) % ACTIONS)
                        : greedy_action(&learner->online, transition.state);
```

如果随机数小于 $\varepsilon$，执行随机动作；否则调用 `greedy_action`。`snake_uniform` 返回 $[0,1)$ 的数，因此该分支以 $\varepsilon$ 的概率发生。选择随机动作时再从 0、1、2 中取一个编号。

所以 $n=0$ 时 $\varepsilon=1$，$n=25000$ 时为 $0.525$，从 $n=50000$ 起保持 $0.05$。若三个随机动作等概率且最大 Q 动作唯一，选中贪心动作的总概率为 $1-\varepsilon+\varepsilon/3$；其他每个动作概率为 $\varepsilon/3$。评估和演示时 $\varepsilon=0$，只计算网络输出并选最大 Q：

当训练到 $n=25000$ 时，$\varepsilon=0.525$，贪心动作的总概率是 $1-0.525+0.525/3=0.65$；另外两个动作各有 $0.525/3=0.175$ 的概率。到 $n=50000$ 后，$\varepsilon=0.05$，贪心动作概率为 $0.95+0.05/3\approx0.966667$，每个非贪心动作概率约为 $0.016667$。这里的贪心动作概率包含“随机探索时恰好抽中它”的机会。

$$
a_t=\operatorname{argmax}_{a\in\{0,1,2\}}Q_\theta(x_t,a).
$$

`greedy_action` 从动作 0 开始作为当前最佳值，再逐个比较剩余 Q 值：

```c
int best = 0;
for (int action = 1; action < ACTIONS; ++action) {
    if (cache.q[action] > cache.q[best]) {
        best = action;
    }
}
return best;
```

严格大于号意味着 Q 值相同时保留较小的动作编号。推理时不执行探索分支，也不更新网络参数。

`eval` 和 `play` 不计算目标、不求梯度、不更新参数。它们使用保存的 5123 个参数反复执行“观测 → 前向传播 → 选动作”。模型文件不含 Adam 历史和经验池，因而用于推理而非完整恢复训练。

### 11. 用多局统计判断模型表现

一局得分是最终蛇长减初始长度 3。对 $N$ 局得分 $S_1,\dots,S_N$，报告经验平均分、最高分和截断比例：

$$
\overline S=\frac{1}{N}\sum_{i=1}^{N}S_i,
\qquad S_{\max}=\max_i S_i,
\qquad p_{\text{truncated}}=
\frac{\text{截断局数}}{N}.
$$

这里 $N$ 是评估局数，$S_i$ 是第 $i$ 局得分，$i$ 从 1 数到 $N$；$S_{\max}$ 是这组分数中的最大值，$p_{\text{truncated}}$ 是达到步数上限的局数占比。

例如 3 局得分是 $2,4,0$，其中第 2 局因步数上限截断，则平均分是 $(2+4+0)/3=2$，最高分是 $\max(2,4,0)=4$，截断比例是 $1/3\approx33.3\%$。

在 `evaluate` 的每局循环内，程序把分数加到 `total`，用比较更新最高分；所有局结束后再除以局数得到平均值：

```c
/* 每局结束时： */
size_t score = game.length - 3;
total += (double)score;
if (score > result.maximum) {
    result.maximum = score;
}
result.truncated += !terminated;

/* 全部局结束后： */
result.mean = total / (double)games;
```

`result.truncated` 累加截断局数，`result.mean` 对应 $\overline S$，`result.maximum` 对应 $S_{\max}$。训练过程用固定验证集挑选模型，最终测试集只汇报结果；相关流程位于 [`evaluate`](../src/games/snake_ai.c)。

平均分概括这些测试局面的总体表现，最高分只代表其中最好的一局；二者都不等于所有可能食物序列上的真实期望。训练每 200 局使用固定 30 局验证集选择 `best.model`，最后用独立的 100 局测试集报告结果。若反复根据测试分数调整模型，测试集就不再是独立评估。棋盘有 $20\times10=200$ 格，蛇初始长度 3，因此理论最高分为 $200-3=197$；训练算法不保证达到它。

## 记录、模型和评估

每次训练创建新目录，不覆盖旧实验。模型、配置和 CSV 在 `sessions/snake-ai/<编号>/`，通过日志记录器写出的 `train.log` 和 `report.txt` 在 `logs/snake-ai/<编号>/`。日志每行带本地毫秒时间戳及调用文件名、行号，旧实验日志不会迁移或修改。产物包括：

- `config.txt`：实际种子、预算、超参数和数据划分。
- `train.log`：每局记录、定期验证和模型更新；终端仅每 20 局显示训练摘要。
- `metrics.csv`：每局步数、分数、奖励、探索率、损失、更新数、结束/截断状态及时间。
- `best.model`：按验证集平均分选择的最佳模型，同分保留此前模型。
- `final.model`：最后一组参数，可能弱于最佳模型。
- `report.txt`：随机策略、未训练网络和最佳模型在独立测试集上的对照结果。

### 如何读懂训练记录

`metrics.csv` 一行对应一局：`episode` 是局号，`steps` 是这一局走了几步，`score` 是吃到的食物数，`reward` 是本局每步总奖励之和，`epsilon` 是当时随机探索概率，`updates` 是累计参数更新次数，`elapsed` 是训练开始后的秒数。`loss` 是最近一次训练批次的平均 Huber 损失；预热前没有参数更新，它可能仍为初始值。`terminated=1` 表示死亡或获胜，`truncated=1` 表示达到本局 4000 步；训练预算在局中用完时，两列都可以是 0。

```sh
head -n 6 sessions/snake-ai/run-XXXXXX/metrics.csv
tail -n 5 sessions/snake-ai/run-XXXXXX/metrics.csv
less logs/snake-ai/run-XXXXXX/train.log
```

把路径中的编号换成实际运行编号。终端每 20 局才打印一次训练摘要，`train.log` 会保存每局及验证信息；日志路径始终位于 `logs/`。单局高分和单批次低损失都容易波动，应重点观察固定验证集的**平均分**是否随训练改善。损失变低表示 Q 预测更接近当前的 TD 目标，不直接等于“游戏得分一定变高”。

每 200 局在固定 30 个验证种子上评估，结束后再验证一次。最终用另一组固定 100 个测试种子，报告平均分、最高分、截断比例和满分局数；测试结果不参与参数更新或模型选择。

这是合成游戏的单次训练实验，不能把偶然最高分视为稳定水平。需通过多个训练种子、更多未见食物序列评估泛化。随机基线只在三个合法转向动作中随机采样，也没有额外避险。

模型文本包含格式、特征、网络尺寸和棋盘版本，数值以 17 位精度写入。先写新临时文件再替换目标文件；加载拒绝版本不匹配、缺失参数、非有限数值和多余内容，失败不会修改已有网络。只保存推理参数，第一版不支持恢复 Adam 和经验池继续训练。

训练、验证和测试各有分工：训练经验更新权重；固定的 30 局验证集只决定何时保存 `best.model`；独立的 100 局测试集只在最后报告表现。`final.model` 是训练结束时的参数，`best.model` 是验证平均分最高时的参数，后者也可能仍是训练前的初始网络。`eval` 和 `play` 只加载网络并选动作，不调用 `learn_batch`，因此观看游戏不会让模型继续学习。若要比较改动效果，应固定测试方法并尝试多个训练种子；单次测试最高分只说明这 100 局里出现过的最好结果。

## 检查与扩展练习

### 建议按这五步亲自验证

1. **只看规则。** 阅读 [`snake_env.h`](../src/games/snake_env.h) 的 `snake_reset → snake_direction → snake_collision → step → snake_act`。在纸上画一个蛇头朝右的局面，分别写出动作 0/1/2 的方向，预测前方是墙、食物或旧尾格时的结果。再看 `tests/snake_ai_test.c` 的 `test_environment` 核对。
2. **手工编码观测。** 根据一个自画的棋盘填写 11 个特征，再按 `snake_observe` 的赋值顺序逐项检查。改变食物到蛇头的相对位置，确认方位位如何变化；改变身体，确认危险位如何变化。
3. **跟踪一次预测和学习。** 在 VS Code 选择 `C: AI 短训练`，先在 `train` 调用 `snake_act` 和 `replay_push` 的位置设断点，观察 `transition.state/action/reward/next/terminated`。再在 `learn_batch`、`td_target`、`network_gradient`、`adam_update` 处设断点，观察同一批次如何得到目标、损失和参数更新。调试配置限制为 1500 步或 30 秒，适合观察流程；`C: AI 单元测试` 适合专看数值函数。
4. **核对梯度而非猜测。** 阅读 `tests/snake_ai_test.c` 的 `test_gradient`：它给某个权重加上、减去微小扰动，比较损失变化算出的数值梯度与链式法则的解析梯度。再看 `test_replay` 如何检查终止目标、截断后继续估值、经验池覆盖和第 500 次更新时的目标网络同步。
5. **比较策略。** 用 `eval` 测试同一模型，查看 `report.txt` 中随机策略、未训练网络、最佳模型在相同测试种子列表上的平均分。改变 `--seed` 再评估，观察最高分和平均分怎样变化；不要根据这次新的测试结果回头挑模型，否则测试集就不再独立。

源码分工可作为阅读地图：[`snake_env.h`](../src/games/snake_env.h) 负责规则和特征，[`snake_ai_network.h`](../src/games/snake_ai_network.h) 负责网络、梯度、Adam 和模型读写，[`snake_ai.c`](../src/games/snake_ai.c) 负责经验池、训练、验证与演示，[`snake_terminal.h`](../src/games/snake_terminal.h) 只处理终端显示和得分。遇到不熟悉的 C 语法，可先查 [`01_c_basics.md`](01_c_basics.md)，再回来追数据流。

### 本机实测结果

Apple Clang 17、CPU、种子 42，实际训练 180 秒（局数上限提高为 100000），完成 35630 局、5244800 环境步和 1310951 次参数更新。训练过程中最高分为 45；验证集选择的最佳模型在独立 100 局测试集上的结果为：

| 策略 | 平均分 | 最高分 | 截断比例 | 满分局数 |
| --- | --- | --- | --- | --- |
| 随机策略 | 0.15 | 1 | 0% | 0 |
| 未训练网络 | 0.02 | 1 | 18% | 0 |
| 最佳训练模型 | 23.53 | 50 | 0% | 0 |

这是一次实验结果，不是所有种子或机器上的性能保证。训练过程中带探索，测试时不探索，因此测试最高分可以高于训练最高分。

这组历史实验原位于 `sessions/snake-ai/run-xrDGfP/`。实验目录被 Git 忽略，不随仓库分发；若本机没有该模型，请先训练，再用本次实际输出的运行编号观看：

```sh
./build/games/snake_ai play --model sessions/snake-ai/run-XXXXXX/best.model
```

另一轮默认局数上限的历史实验在约 96 秒达到 20000 局，原运行编号为 `run-bE7m7m`；两次训练均独立创建目录。版本控制只包含源码和本节结果说明。

### 验证命令

```sh
make test                 # 现有示例、人工游戏和 AI 单元测试
make lint
make test-snake-terminal  # 人工终端回归
make test-snake-ai        # 数值测试、短训练复现及 AI 终端测试
make sanitize-snake-ai    # 地址与未定义行为检查
```

AI 集成测试需要 Python 3 标准库，不修改用户历史记录。固定步数和种子在同一工具链上可重复；时间预算模式受机器负载影响，不能要求最终参数完全一致。

可逐步尝试：增加食物距离和射线特征、完整棋盘输入、Double DQN、训练恢复、多个种子统计以及基于规则的对照策略。简化观测和有限训练不能保证达到 197 分；优化目标是不断提高独立评估的平均分和最高分。

算法参考：[Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236)。
