# 类别不平衡与概率校准：源码分析

本文跟着 [`src/ml/24_reliability_assessment.c`](../src/ml/24_reliability_assessment.c) 的实际数据流，学习正类稀少时如何评估二分类模型。案例从固定种子生成合成数据，比较多数类基线、普通损失与类别加权损失；随后只用验证集选决策阈值并拟合 Platt 校准映射，最后在测试集报告分类指标和概率质量。运行 `make reliability-assessment` 查看完整表格及 ASCII 可靠性图。

这是一个只有 $400$ 条样本的教学实验，不代表真实应用中的性能。验证集和测试集各只有 $8$ 个正例，指标会随样本变化而明显波动；它们的用途是理解计算和数据隔离。

![预测概率与实际正类频率的可靠性示意图](images/44_reliability.svg)

虚线代表概率校准理想状态：预测概率与实际频率相等。绿线示意按概率分箱后的偏离；真实评估需使用测试结果计算。

## 1. 输入、输出和源码地图

程序模拟“少数类事件识别”：输入是两个数值特征，标签 $y_i=1$ 表示稀少的正类，$y_i=0$ 表示负类。程序输出离散分类结果的混淆矩阵和指标，也输出概率预测的 Brier 分数、交叉熵、ECE 与可靠性图。

| 概念 | 源码位置 | 代码字段或变量 | 作用 |
| --- | --- | --- | --- |
| 划分枚举 | [`DataSplit`](../src/ml/24_reliability_assessment.c#L25) | `SPLIT_TRAIN`、`SPLIT_VALIDATION`、`SPLIT_TEST` | 区分训练、验证和测试数据。 |
| 样本与计数 | [`Sample`、`Dataset`](../src/ml/24_reliability_assessment.c#L31) | `features`、`label`、`split_counts`、`class_counts` | 保存特征、真实类别及每个划分的类别比例。 |
| 确定性合成数据 | [`dataset_generate`](../src/ml/24_reliability_assessment.c#L120) | `negatives`、`positives`、`seed` | 生成两类样本并分别随机化，再分层切出训练/验证/测试集。 |
| 训练集标准化 | [`standardizer_fit`](../src/ml/24_reliability_assessment.c#L226) | `mean`、`scale`、`training_count` | 只由训练特征计算均值和总体标准差。 |
| 模型和预测值 | [`LogisticModel`](../src/ml/24_reliability_assessment.c#L49)、[`Prediction`](../src/ml/24_reliability_assessment.c#L59) | `weights`、`bias`、`logit`、`probability` | 关联逻辑回归参数、原始分数和正类概率。 |
| 普通/加权训练 | [`model_train`](../src/ml/24_reliability_assessment.c#L370) | `balance_classes`、`class_weights`、`weight_gradients` | 同一梯度下降代码可训练普通模型或类别加权模型。 |
| 混淆矩阵与指标类型 | [`Metrics`](../src/ml/24_reliability_assessment.c#L65) | `true_positive`、`false_negative` 等 | 保存计数及派生指标。 |
| 混淆矩阵计算 | [`classification_metrics`](../src/ml/24_reliability_assessment.c#L475) | `threshold`、`true_positive`、`false_negative` 等 | 将连续概率按阈值转成类别，并累计指标。 |
| 阈值选择 | [`select_threshold`](../src/ml/24_reliability_assessment.c#L533) | `candidate`、`best_metrics` | 在验证集网格上最大化 F1，平局时用平衡准确率决胜。 |
| Platt 映射拟合与应用 | [`calibrator_fit`](../src/ml/24_reliability_assessment.c#L576)、[`calibrator_apply`](../src/ml/24_reliability_assessment.c#L687) | `slope`、`intercept`、`calibrated_logit` | 在验证集拟合仿射 logit 映射，并应用到未参与拟合的测试预测。 |
| 概率指标与分箱 | [`probability_metrics`](../src/ml/24_reliability_assessment.c#L625)、[`print_probability_metrics`](../src/ml/24_reliability_assessment.c#L769) | `brier_score`、`cross_entropy`、`ReliabilityBin` | 计算概率误差、十箱 ECE，并绘制 ASCII 可靠性图。 |
| 完整执行顺序 | [`main`](../src/ml/24_reliability_assessment.c#L829) | `dataset`、`validation_predictions`、`test_predictions` | 训练、选阈值、拟合校准器，最后才打开测试指标路径。 |

## 2. 生成数据和隔离三个阶段

`dataset_generate` 生成 $360$ 个负类和 $40$ 个正类，总体正类比例为：

$$
\pi=\frac{40}{360+40}=0.1=10\%.
\tag{1}
$$

这里 $\pi$ 表示正类先验比例。负类特征分别来自区间 $[-1,1)$；正类的两个特征分别在这个范围上平移。固定随机种子 `20261021` 让样本与输出可重复。源码先分别打乱两个类别，再为每一类按 $60\%/20\%/20\%$ 划分训练、验证、测试数据，因此每个划分保留同样的类别比例：

| 数据划分 | 负类 $y=0$ | 正类 $y=1$ | 总数 |
| --- | ---: | ---: | ---: |
| 训练集 | $216$ | $24$ | $240$ |
| 验证集 | $72$ | $8$ | $80$ |
| 测试集 | $72$ | $8$ | $80$ |

源码中的 `DataSplit` 和 `Dataset.class_counts[split][label]` 保存这些状态。`standardizer_fit` 和 `model_train` 都只筛选 `SPLIT_TRAIN`；`select_threshold` 与 `calibrator_fit` 只接收验证集预测。测试预测收集发生在两项验证集决策之后，测试标签没有进入梯度、阈值搜索或校准拟合。

训练标准化使用训练集均值和总体标准差，形式与前一案例一致：

$$
\tilde{x}_{ij}=\frac{x_{ij}-\mu_j}{s_j}.
\tag{2}
$$

$i$ 是从 $0$ 开始的样本下标，$j\in\{0,1\}$ 是两个特征的下标，$x_{ij}$ 是原始特征，$\mu_j$ 和 $s_j$ 分别对应 `Standardizer.mean[j]`、`Standardizer.scale[j]`，$\tilde{x}_{ij}$ 是标准化结果。训练、验证、测试都复用同一组训练统计量，避免测试信息泄漏。

## 3. 为什么准确率会掩盖少数类失败

混淆矩阵把真实标签与预测标签交叉统计。以正类为目标：

- $TP$：真实为正且预测为正；
- $FP$：真实为负却预测为正；
- $FN$：真实为正却预测为负；
- $TN$：真实为负且预测为负。

源码的 `Metrics.true_positive`、`false_positive`、`false_negative`、`true_negative` 分别存放这四个整数，判断逻辑在 [`classification_metrics`](../src/ml/24_reliability_assessment.c#L475)。准确率为：

$$
\mathrm{Accuracy}=\frac{TP+TN}{TP+TN+FP+FN}.
\tag{3}
$$

精确率回答“预测成正类的样本里有多少是真的正类”；召回率回答“所有真实正类中找回了多少”：

$$
\mathrm{Precision}=\frac{TP}{TP+FP},
\qquad
\mathrm{Recall}=\frac{TP}{TP+FN}.
\tag{4}
$$

F1 是精确率和召回率的调和平均；平衡准确率把正类召回率与负类召回率（特异度）同等看待：

$$
\mathrm{F1}=\frac{2\,\mathrm{Precision}\,\mathrm{Recall}}
{\mathrm{Precision}+\mathrm{Recall}},
\qquad
\mathrm{BalancedAccuracy}=\frac{1}{2}
\left(\frac{TP}{TP+FN}+\frac{TN}{TN+FP}\right).
\tag{5}
$$

例如测试集有 $72$ 个负类和 $8$ 个正类。始终预测负类的多数类基线得到 $TN=72$、$FN=8$、$FP=0$、$TP=0$。因此：

$$
\mathrm{Accuracy}=\frac{72}{80}=0.9,
\qquad
\mathrm{Recall}=\frac{0}{8}=0,
\qquad
\mathrm{BalancedAccuracy}=\frac{0+1}{2}=0.5.
\tag{6}
$$

它看起来有 $90\%$ 准确率，却没有找到任何正例。源码对零分母统一返回 $0$：没有预测正类时精确率为 $0$，没有真实正类时召回率为 $0$；这样可以避免 `NaN`，但使用者仍应结合混淆矩阵解释指标。

## 4. 类别加权损失与一次梯度贡献

普通二元交叉熵把每个样本看得同样重要。当负类远多于正类时，模型可以主要优化负类。案例另外训练类别加权模型，给训练集中的类别 $c$ 使用权重：

$$
\alpha_c=\frac{N_{\mathrm{train}}}{2N_c}.
\tag{7}
$$

$N_{\mathrm{train}}=240$ 是训练样本总数，$N_c$ 是训练集中类别 $c$ 的样本数，$\alpha_c$ 对应 `class_weights[c]`。代入 $N_0=216$、$N_1=24$：

$$
\alpha_0=\frac{240}{2\times216}=\frac{5}{9}\approx0.556,
\qquad
\alpha_1=\frac{240}{2\times24}=5.
\tag{8}
$$

所以单个正例在损失中约相当于 $9$ 个负例。源码的 `balance_classes` 为 `false` 时两类权重都是 $1$，为 `true` 时使用上式；两次训练共用同一批训练样本、标准化参数和 $1200$ 轮全批量梯度下降。

令 $z_i=b+\sum_jw_j\tilde{x}_{ij}$ 为逻辑回归 logit，$p_i=\sigma(z_i)$ 为正类概率，$y_i\in\{0,1\}$ 为真实标签。单样本二元交叉熵及加权平均目标为：

$$
\ell_i=-y_i\log p_i-(1-y_i)\log(1-p_i),
\qquad
L=\frac{1}{N_{\mathrm{train}}}\sum_{i\in\mathcal{I}_{\mathrm{train}}}
\alpha_{y_i}\ell_i.
\tag{9}
$$

这里 $\mathcal{I}_{\mathrm{train}}$ 是训练样本下标集合，$w_j$ 和 $b$ 是模型权重与偏置，$\sigma$ 是 sigmoid。下面梯度中的求和都遍历训练样本。对 sigmoid 与交叉熵求导，logit 梯度化简成 $p_i-y_i$，类别权重再乘到误差上：

$$
\frac{\partial L}{\partial z_i}
=\frac{\alpha_{y_i}}{N_{\mathrm{train}}}(p_i-y_i),
\qquad
\frac{\partial L}{\partial w_j}
=\frac{1}{N_{\mathrm{train}}}\sum_i\alpha_{y_i}(p_i-y_i)\tilde{x}_{ij},
\qquad
\frac{\partial L}{\partial b}
=\frac{1}{N_{\mathrm{train}}}\sum_i\alpha_{y_i}(p_i-y_i).
\tag{10}
$$

源码对应 `error`、`weight_gradients`、`bias_gradient` 以及后面的参数更新，见 [`model_train`](../src/ml/24_reliability_assessment.c#L370)。为手算一次单样本的局部梯度，假设初始化时 $z=0$、因此 $p=0.5$；样本标签 $y=1$，类别权重 $\alpha_1=5$，第 $0$ 个标准化特征为 $\tilde{x}_0=0.5$。这个样本给权重梯度和偏置梯度的未平均贡献是：

$$
g_{w_0}^{(i)}=5(0.5-1)(0.5)=-1.25,
\qquad
g_b^{(i)}=5(0.5-1)=-2.5.
\tag{11}
$$

上标 $(i)$ 表示第 $i$ 个样本贡献，不是乘方。完整批次先累加所有贡献并除以 $N_{\mathrm{train}}$，再用学习率 $\eta=0.1$ 更新参数：

$$
w_j\leftarrow w_j-\eta\frac{1}{N_{\mathrm{train}}}
\sum_i\alpha_{y_i}(p_i-y_i)\tilde{x}_{ij},
\qquad
b\leftarrow b-\eta\frac{1}{N_{\mathrm{train}}}
\sum_i\alpha_{y_i}(p_i-y_i).
\tag{12}
$$

只看上述一个样本在 $w_0$ 上的平均梯度份额是 $-1.25/240\approx-0.005208$；若只考虑该份额，它使 $w_0$ 增加约 $0.000521$。实际程序会使用整个训练批次的梯度。该代入说明了权重机制：相同预测错误的正类对梯度影响更大。固定种子下，测试集的加权模型在阈值 $0.5$ 时得到 $TP=4$、$FP=6$、$FN=4$、$TN=66$，召回率从多数类基线的 $0$ 增至 $0.5$，但准确率从 $0.9$ 降到 $0.875$。类别加权是精确率/召回率权衡手段，不保证每项指标同时上升。

## 5. 阈值只在验证集选择

模型输出概率 $p_i$，分类还需要一个决策阈值 $\tau$：

$$
\hat{y}_i=
\begin{cases}
1,&p_i\geq\tau,\\
0,&p_i<\tau.
\end{cases}
\tag{13}
$$

$\hat{y}_i$ 是预测类别，$\tau$ 对应 `classification_metrics` 的 `threshold`。阈值增大时更难预测成正类，通常会减少假阳性并可能增加假阴性。`select_threshold` 在验证概率上扫描 $0.05$ 到 $0.95$、步长 $0.01$ 的候选值，先最大化 F1；F1 平局时优先平衡准确率，再平局则取较高阈值。固定种子的验证集选择出原始阈值 $\tau_{\mathrm{raw}}=0.84$，验证集 F1 为 $0.545$。在测试集上该阈值得到 $TP=2$、$FP=0$、$FN=6$、$TN=72$：

$$
\mathrm{Precision}=\frac{2}{2+0}=1,
\qquad
\mathrm{Recall}=\frac{2}{2+6}=0.25,
\qquad
\mathrm{F1}=\frac{2\times1\times0.25}{1+0.25}=0.4.
\tag{14}
$$

不能根据这组测试结果再调低阈值；那会把测试集变成调参数据。阈值选择还依赖业务中误报和漏报的代价。本案例用 F1 做教学目标，并不表示它是所有应用的正确指标。

## 6. Platt 映射：让概率更贴近频率

分类阈值只把概率变成类别，不能回答“模型说 $70\%$ 时，这类样本是否真的约有 $70\%$ 为正”。校准通过验证集拟合一个新映射，让概率数值尽量与观测频率一致。代码使用 Platt 仿射映射：

$$
q_i=\sigma(a z_i+c).
\tag{15}
$$

$z_i$ 是原始模型 logit，$q_i$ 是校准概率，$a$ 对应 `CalibrationMap.slope`，$c$ 对应 `CalibrationMap.intercept`。拟合时最小化验证集平均二元交叉熵。令 $r_i=a z_i+c$，单个验证样本的误差项为 $\sigma(r_i)-y_i$，链式法则得到：

$$
\frac{\partial L_{\mathrm{val}}}{\partial a}
=\frac{1}{N_{\mathrm{val}}}\sum_i(\sigma(a z_i+c)-y_i)z_i,
\qquad
\frac{\partial L_{\mathrm{val}}}{\partial c}
=\frac{1}{N_{\mathrm{val}}}\sum_i(\sigma(a z_i+c)-y_i).
\tag{16}
$$

$L_{\mathrm{val}}$ 是验证集平均交叉熵，$N_{\mathrm{val}}=80$ 是验证样本数。`calibrator_fit` 从 $a=1,c=0$ 开始做 $2000$ 轮梯度下降，学习率为 $0.03$；梯度按验证样本数取平均。固定种子拟合出 $a\approx1.0225$、$c\approx-1.9313$。验证集 BCE 从 $0.4088$ 降至 $0.2303$。映射拟合完成后，`calibrator_apply` 才将同一组参数应用到测试 logits。

以程序报告的首个测试正例为例，原始 logit 为 $z=0.0335$，原始概率为 $\sigma(0.0335)\approx0.5084$。代入拟合参数：

$$
r=1.0225\times0.0335-1.9313\approx-1.8971,
\qquad
q=\sigma(-1.8971)\approx0.1304.
\tag{17}
$$

这个单个正例的校准概率变低，不代表映射失效：校准优化的是整组验证样本的概率与频率关系，不是强迫每个正样本都给高分。测试集的正类比例是 $10\%$，映射会把加权训练导致的整体偏高概率往低处调整。概率校准关注概率质量；它不会自动提高分类准确率。

本例斜率 $a$ 为正，因此映射保持样本排序。校准前后分别在验证集选择阈值，得到 $0.84$ 与 $0.46$，测试混淆矩阵仍相同（$TN=72,FP=0,FN=6,TP=2$）。可通过反函数理解两个阈值如何对应：校准概率阈值 $\tau_q$ 等价于原始概率阈值

$$
\tau_p=\sigma\left(\frac{\operatorname{logit}(\tau_q)-c}{a}\right),
\qquad
\operatorname{logit}(u)=\log\frac{u}{1-u}.
\tag{18}
$$

将 $\tau_q=0.46$、$a=1.0225$、$c=-1.9313$ 代入，得到 $\tau_p\approx0.85$，与网格选择的 $0.84$ 很接近。因此两个阈值在此数据上划分出相同类别；校准改变的是概率刻度，而不是样本排序。

## 7. Brier、交叉熵和可靠性图

Brier 分数是概率与二元标签之间的均方误差。分数越低表示平方概率误差越小：

$$
\mathrm{Brier}=\frac{1}{N}\sum_{i=1}^{N}(p_i-y_i)^2.
\tag{19}
$$

这里 $N$ 是待评估划分的样本数，$p_i$ 是当前概率；校准前 $p_i$ 是 `Prediction.probability`，校准后是 $q_i$。源码通过 logit $z_i$ 计算 BCE，校准后 $z_i$ 表示 $a z_i+c$。平均 BCE 与稳定的单样本表达式为：

$$
\mathrm{BCE}=\frac{1}{N}\sum_{i=1}^{N}\ell_i,
\qquad
\ell_i=\max(z_i,0)-y_i z_i+\log\left(1+e^{-|z_i|}\right).
\tag{20}
$$

这个形式避免直接计算概率接近 $0$ 或 $1$ 时的 $\log(0)$，见 [`binary_cross_entropy`](../src/ml/24_reliability_assessment.c#L319)。例如前文的测试正例 $y=1$，校准前 $p=0.5084$，单样本 Brier 贡献约为 $(0.5084-1)^2=0.2417$；校准后 $q=0.1304$，该正例贡献变成 $(0.1304-1)^2\approx0.7562$。单个样本变差而整体测试集 Brier 从 $0.1129$ 降至 $0.0735$ 并不矛盾，校准比较的是一组样本的总体概率误差。

可靠性图把概率分成十个等宽箱。对箱 $b$，令 $n_b$ 为样本数，$\bar{p}_b$ 为平均预测概率，$o_b$ 为实际正例比例；ECE 将箱内概率差按箱大小加权：

$$
\mathrm{ECE}=\sum_{b=1}^{10}\frac{n_b}{N}
\left|\bar{p}_b-o_b\right|.
\tag{21}
$$

以测试集校准前 $[0.0,0.1)$ 区间为例，程序统计 $n_b=32$、$\bar{p}_b\approx0.033$、$o_b\approx0.062$。该箱对 ECE 的贡献约为：

$$
\frac{32}{80}\left|0.033-0.062\right|
=0.4\times0.029\approx0.0116.
\tag{22}
$$

其余非空箱按相同方式累加。`ReliabilityBin` 记录样本数、概率和与正例数；输出表格直接显示每箱平均概率与实际正例率。ASCII 图上的 `.` 是理想对角线，`o` 是每个非空箱的观测点，重合时用 `*`。

固定种子测试结果如下：

| 概率评估 | Brier | BCE | ECE |
| --- | ---: | ---: | ---: |
| 校准前测试概率 | $0.1129$ | $0.3627$ | $0.1627$ |
| 校准后测试概率 | $0.0735$ | $0.2921$ | $0.0673$ |

在本次固定合成划分上，三个数值都下降了；这只是一个教学观察。ECE 依赖分箱数和分箱边界，少样本时波动较大；校准改善验证集损失也不保证测试集一定改善，更不保证新分布上仍然校准。

## 8. 失败边界与测试覆盖

- `dataset_generate` 使用固定容量数组和可复现种子；输出通过局部对象构造后整体赋值。它不读取外部文件，也不依赖第三方数据处理库。
- 标准化参数只从训练样本拟合。非法划分、非有限特征或零方差会失败；函数不提交部分统计量。
- `classification_metrics` 拒绝空样本、非有限/越界概率以及不在 $[0,1]$ 内的阈值。精确率、召回率的分母为零时按代码约定返回 $0$。
- `select_threshold` 使用固定网格而不是连续优化。网格之外的阈值不会被选择；平局规则明确为 F1、平衡准确率、较高阈值的先后顺序。
- `calibrator_fit` 要求验证集同时含正负类。若缺少一类或 logit 无效，拟合失败并保留原映射。此案例不对斜率加正值约束；当前固定数据拟合得到正斜率，保持排序。
- 十箱图表可能包含空箱；空箱显示为 `-`，ECE 只累加非空箱。Brier、BCE 和 ECE 都是划分上的汇总，不取代逐样本检查。

测试 [`test_reproducible_pipeline_and_split_isolation`](../tests/reliability_assessment_test.c#L10) 验证固定种子与类别计数可复现，并极端修改测试集特征和标签后确认训练统计量、模型参数不变。手算测试覆盖混淆矩阵、Brier 和零分母；另外验证阈值非法时不改输出、校准验证集缺少类别时失败并保留已有参数。运行：

```sh
make reliability-assessment
make test-reliability-assessment
make sanitize-reliability-assessment
```

这些测试证明固定输入下的实现、边界和数据隔离行为，不证明算法在真实数据上的泛化能力，也不证明少数类业务成本已被正确建模。
