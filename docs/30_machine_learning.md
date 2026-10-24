# 用 C 学习简单机器学习

这里的案例使用标准 C 与数学库，不依赖机器学习框架。它们采用固定、小型合成数据，目的是看清算法步骤，并练习数组、结构体、循环、函数和数值计算。

阅读各专题的公式时，可先查阅[数学公式与符号读法参考](31_reference.md)，了解常见字母、上下标、运算符和公式的读法；每篇教程会再定义本例中的具体含义。

```sh
make ml                              # 运行全部机器学习案例
make build/ml/01_linear_regression    # 只编译一个案例
./build/ml/01_linear_regression
```

数学库通过 `-lm` 链接；更换编译器或参数前先 `make clean`。全部案例也参与 `make test` 与 `make lint`。

![机器学习从数据输入到独立评估的通用流程](images/30_machine_learning.svg)

这张总览图把后续案例放到同一条学习线上：明确输入与任务、理解模型输出，再用独立数据检查效果。聚类没有训练标签。

## 先理解几个词

- **样本**：一条观测数据，例如一个二维点。
- **特征**：用于预测的输入，例如点的横纵坐标。
- **标签**：已知答案，例如类别 0 或 1。
- **训练**：用数据确定参数或保存用于预测的数据。
- **预测**：把新输入交给已确定的模型或规则。
- **损失**：衡量当前预测偏离训练目标的程度。

回归输出数值，分类输出类别，聚类在没有标签时寻找分组。线性回归、逻辑回归和 K-means 的基本推导可参考 [Stanford CS229 课程资料](https://cs229.stanford.edu/materials.html-full)。

## 1. 线性回归：学习一条直线

源码：`src/ml/01_linear_regression.c`。

### 查看完整训练日志

示例使用 `src/common/logger.h` 中的单线程日志记录器，同时向屏幕和文件输出全部训练过程。默认文件为 `logs/linear_regression.log`，采用追加模式。每行格式为 `2026-10-06 02:31:58,019 | 01_linear_regression.c:48 | 内容`，时间使用本地时区，文件名和行号来自日志调用处。每次写入都会刷新；打开、写入、刷新或关闭失败会使程序返回失败状态。

```sh
./build/ml/01_linear_regression
less logs/linear_regression.log                  # 分页查看全部过程，按 q 退出
./build/ml/01_linear_regression logs/training.log # 自定义文件路径
```

日志路径须使用 `logs/` 下的相对路径，父目录自动创建，拒绝 `..` 等目录回退。消息只格式化一次，多行逐行添加相同前缀；两个目标内容一致，没有末尾换行时自动补齐。`logger_write` 宏通过 `__FILE__` 和 `__LINE__` 捕获调用位置。日志记录器不关闭借用的标准输出流；每次刷新适合教学演示，大量生产日志可改用缓冲策略。旧日志原样保留。

数据遵循 `y = 2x + 1`，但程序从权重和偏置均为零开始。模型为 `prediction = weight * x + bias`，损失为平均平方误差：

```text
error = prediction - y
loss = sum(error²) / count
d_weight = 2 * sum(error * x) / count
d_bias = 2 * sum(error) / count
parameter -= learning_rate * gradient
```

每轮使用全部训练数据更新一次参数，属于批量梯度下降。示例迭代 500 轮，学习率为 0.05，最终参数接近 `weight=2`、`bias=1`；输入 3 得到约 7。

验证：损失下降、最终损失接近零、训练数据之外的两个输入预测正确。

逐步推导和源码走读见[线性回归分析](32_linear_regression_analysis.md)。

练习：把关系改为 `y = -3x + 2`，添加小噪声，观察损失不能降到零。增大学习率，观察发散风险。输入尺度增大时，重新考虑标准化与学习率。

## 2. 逻辑回归：输出二分类概率

源码：`src/ml/02_logistic_regression.c`。

训练数据中负数标为 0，正数标为 1。先计算 `z = weight * x + bias`，再用 sigmoid 将结果映射到 0 到 1：

```text
probability = 1 / (1 + exp(-z))
error = probability - label
d_weight = sum(error * x) / count
d_bias = sum(error) / count
```

损失使用二元交叉熵；预测概率达到 0.5 时可判为类别 1。示例使用分支实现 sigmoid，并用 `max(z,0) - label*z + log1p(exp(-abs(z)))` 计算损失，避免直接计算极端指数及 `log(0)`。

验证：交叉熵下降、未训练的正负输入分类正确、极端 sigmoid 输入不溢出。

完整公式推导、数值算例和源码走读见[逻辑回归分析](33_logistic_regression_analysis.md)。

练习：改变分界线，例如以 1 为阈值生成标签；添加冲突样本观察结果。数据完全线性可分时，无正则化的参数可能不断增长，有限迭代的结果不代表存在有限的最优参数。

## 3. K 近邻：用周围样本投票

源码：`src/ml/03_knn.c`。

六个二维样本分成两类。预测时计算新点到每个样本的欧氏距离平方，排序后选最近的 k 个样本，用多数投票决定类别。无需迭代拟合权重，但预测需要访问训练样本。

示例使用结构体存放样本和距离，通过 `qsort` 排序。距离相同时按原始索引排序；票数相同时选类别 0，保证规则明确。k 必须在 1 到样本数之间；坐标及距离计算假设为有限值且不会溢出。

验证：两组新点分类、票数平局、无效 k 及失败时输出不变。

从距离公式开始的完整推导、手算过程和源码走读见[K 近邻分析](34_knn_analysis.md)。

练习：比较 k=1、3、5 的结果。给一个特征乘以 100，观察距离被该维主导，再用训练集的统计量做标准化。此实现固定六个样本，扩展时可改为动态数组及显式样本数。

## 4. K-means：无标签地寻找分组

源码：`src/ml/04_kmeans.c`。

两组二维点没有标签。算法固定两个初始中心，然后交替执行：

1. 分配：每个点加入最近中心对应的簇。
2. 更新：每个中心变成所在簇内点的均值。
3. 判断：中心变化足够小则停止，否则继续，最多 100 次。

目标是降低点到所属中心的距离平方和。空簇保留原中心，避免除以零；距离平局选择编号小的簇。示例得到约 `(0.333,0.333)` 与 `(8.333,8.333)` 两个中心。

验证：每轮目标不增加、中心位置和分组正确、相同点造成空簇时保持有限值。

完整的分配距离、均值推导、目标函数和迭代过程见[K-means 分析](35_kmeans_analysis.md)。

练习：改变初始中心，观察收敛结果；增加第三个簇并修改相关循环。簇编号没有类别意义，换初始化后编号可能互换。迭代上限到达时，工程接口应显式报告未收敛。

## 5. 深度学习数值基础

`src/ml/05_numeric_foundations.c` 对应 [学习计划](50_deep_learning_plan.md) 的阶段 0。运行 `make numeric-foundations`，观察矩阵乘法、转置、逐元素相加，以及两输入线性模型的前向、链式法则和一次参数更新。固定小数据配套手算答案、数值梯度检查、容量与分配失败验证；详见 [数值基础教程](51_numeric_foundations.md)。

## 6. XOR 小网络

阶段 1 的 `src/ml/06_xor_mlp.c` 从单个 sigmoid 神经元推进到 `2→4→1` 的 XOR 小网络，手写 tanh、稳定交叉熵、反向传播与逐样本 SGD。运行 `make xor-mlp`，配合 [XOR 小网络教程](52_xor_mlp.md) 学习；内置测试覆盖全部 17 个参数的数值梯度、一次更新、零初始化和四组预测。

## 7. 二维分类与小批量训练

阶段 2 的 `src/ml/07_mlp_2d.c` 使用 `2→16→16→1` ReLU 网络区分内圆盘和外圆环。它实现打乱下标、小批量平均梯度、SGD/Adam 对比、验证集选模和独立测试评估；`make mlp-2d` 导出训练前、中途及最佳模型的决策边界 BMP，默认测试不生成图片。详见 [二维 MLP 教程](53_mlp_2d.md)。

## 8. 卷积网络识别图形

阶段 3 的 `src/ml/08_cnn_shapes.c` 手写四通道卷积、ReLU、最大池化、展平和 softmax 分类，识别随机位置、尺寸及噪声的灰度图形。`make cnn-shapes` 导出样本与训练前后特征图；内置检查覆盖 631 个参数和 256 个输入像素的梯度及池化平局。详见 [CNN 图形教程](54_cnn_shapes.md)。

## 9. 去噪自编码器

阶段 4 的 `src/ml/09_denoising_autoencoder.c` 实现 `256→32→8→32→256` 去噪网络，训练噪声每轮更新、评估噪声固定，以 MSE 学习干净目标。`make autoencoder` 导出带噪/重建/目标拼图；测试检查全部参数和输入梯度、三图拟合及独立去噪效果，并对照带噪输入与训练集平均图基线。详见 [自编码器教程](55_denoising_autoencoder.md)。

## 10. 字符 RNN

阶段 5 的 `src/ml/10_char_rnn.c` 实现 one-hot→16 维 tanh 状态→softmax 下一字符预测，手写完整 BPTT、时间共享参数梯度和全局梯度裁剪。先拟合 `abcabc…`，再用完整独立序列划分的微型字符语料训练、验证和测试，对照只依赖当前字符的 bigram 基线。

运行 `make char-rnn`，`make test-char-rnn` 验证全部参数及初始状态梯度、数值边界和训练效果；可用 `--log` 同时追加到 `logs/ml/10_char_rnn.log`。详见 [字符 RNN 教程](56_char_rnn.md)，其中解释 teacher forcing、生成、验证检查点和过拟合。

## 11. GRU 记忆比特

阶段 6 的 `src/ml/11_gru_memory.c` 用开始比特、无关噪声和结束标记学习序列记忆，只在终点监督。实现 reset-before GRU 的更新门、重置门、候选状态及完整 BPTT，并与同宽 RNN 在相同300轮数据预算下比较三种间隔和两个初始化种子。

运行 `make gru-memory`，`make test-gru-memory` 和 `make sanitize-gru-memory` 验证梯度、边界与训练；`--log` 追加双输出日志至 `logs/ml/11_gru_memory.log`。详见 [GRU 记忆比特教程](57_gru_memory.md)，包括门控公式、种子敏感性和公平比较的边界。LSTM 尚未实现，保留为选做。

## 12. 小型 Transformer

阶段 7 的 `src/ml/12_tiny_transformer.c` 实现16维、双头、上下文16的单层decoder-only模型，包含因果掩码、可学习位置、pre-LayerNorm、残差和GELU。手写全部反向传播，先微型拟合，再复用字符语料的完整训练/验证/测试分组，并展示两头注意力矩阵。

运行 `make tiny-transformer`，用 `make test-tiny-transformer` 和 `make sanitize-tiny-transformer` 验证全部参数、局部梯度、因果性和训练；`--log` 双输出至 `logs/ml/12_tiny_transformer.log`。详见 [小型 Transformer 教程](58_tiny_transformer.md)，包括数据分块、过拟合、贪心生成和上下文限制。

## 13. VAE 图形生成

`src/ml/13_vae_shapes.c` 将16×16图形编码成二维均值/对数方差，用重参数化采样后解码。优化每图Bernoulli重建负对数似然加标准正态KL，固定噪声验证全部参数及输入/epsilon导数，按独立验证负ELBO选择检查点。

运行 `make vae` 将重建对照、潜在网格、潜在插值和先验样本导出到 `exports/`；`make test-vae` 不写文件。`--log` 双输出至 `logs/ml/13_vae_shapes.log`。详见 [VAE 教程](59_vae.md)，其中区分随机重建、解码均值、先验采样及生成质量的评估边界。

## 14. GAN 八簇二维点

`src/ml/14_gan_2d.c` 用两个独立的两层tanh网络学习八个高斯簇。判别器优化真实/生成BCE，生成器优化非饱和损失；生成器更新时冻结判别器参数，但保留其输入导数。没有重建配对或VAE编码器。

运行 `make gan` 导出前后散点、判别器热图与潜在网格；`make test-gan` 和 `make sanitize-gan` 检查完整梯度链、每步冻结与数值边界。`--log` 双输出至 `logs/ml/14_gan_2d.log`。详见 [GAN 教程](60_gan_2d.md)，通过最近簇距离、有效点比例和每簇计数识别模式崩塌，不以D接近0.5作为生成成功的唯一证据。

## 从教学案例走向真实数据

阶段 9.1 的 `src/ml/15_naive_bayes.c` 在固定英文短句上演示多项式朴素贝叶斯。运行 `make naive-bayes` 查看两类未归一化对数分数、独立测试结果及多数类基线；`make test-naive-bayes` 核对手算概率、平滑和未登录词。详见 [朴素贝叶斯教程](36_naive_bayes.md)。

阶段 9.2 的 `src/ml/16_decision_tree.c` 用二维网格演示基尼增益、深度与最小叶节点限制。运行 `make decision-tree` 导出树桩和深度 2 决策边界对照图；`make test-decision-tree` 核对手算增益、平局、重复值与失败保留。详见 [决策树教程](37_decision_tree.md)。

阶段 9.3 的 `src/ml/17_random_forest.c` 在带少量训练标签噪声的二维圆形分类任务上实现 bootstrap、有放回抽样、节点随机特征选择、多数投票和袋外评估。运行 `make random-forest` 比较单棵 CART 与多个种子的 31 棵树森林；`make sanitize-random-forest` 检查内存和未定义行为。详见 [随机森林教程](38_random_forest.md)。

阶段 9.4 的 `src/ml/18_pca.c` 使用固定种子的相关四维数据，仅从训练集拟合均值、协方差和两个主方向，再将训练与独立测试样本投影到二维并重建。运行 `make pca` 导出投影散点图；直接运行 `./build/ml/18_pca` 查看方差解释比例及两组重建误差。详见 [PCA 教程](39_pca.md)。

阶段 9.5 的 `src/ml/19_linear_svm.c` 用固定种子的二维数据手写带 L2 正则的软间隔线性 SVM，以小批量次梯度下降优化 hinge 损失，并在相同训练集上拟合逻辑回归作对照。运行 `make linear-svm` 查看训练/测试错误、几何间隔和两种决策边界；直接运行 `./build/ml/19_linear_svm` 只输出数值。详见 [线性 SVM 教程](40_linear_svm.md)。

阶段 10.1 的 `src/ml/20_cross_validation.c` 用分层 4 折交叉验证选择逻辑回归的 L2 系数。每折只从折内训练数据拟合标准化参数；选参后在完整开发集重训，最终测试集只评估一次。运行 `make cross-validation` 查看每折和平均 log-loss、所选系数与最终测试指标。详见 [交叉验证教程](41_cross_validation.md)。

阶段 10.2 的 `src/ml/21_regularization.c` 用 $24$ 条训练样本拟合 $2\to24\to1$ MLP，每条训练标签以 $20\%$ 概率翻转，对照无正则与五种非零 L2 强度。运行 `make regularization` 观察训练 BCE、含惩罚项的优化目标、验证 BCE 和权重范数；验证集选择系数，三个种子的独立测试集只用于最终报告。网络共有 $97$ 个参数；`make test-regularization` 对全部参数做中心差分检查，并验证偏置不受惩罚；`make sanitize-regularization` 运行消毒器。详见 [正则化教程](42_regularization.md)。

进一步学习神经网络和强化学习，可阅读 [纯 C 深度强化学习贪吃蛇](62_snake_ai.md)。它包含两层隐藏层、反向传播、Adam、经验回放、目标网络以及独立评估，实现自动玩本项目的贪吃蛇。

前几个案例主要验证算法实现与已知小数据上的行为；二维分类、CNN 和自编码器案例已拆分训练、验证和测试集，但一次固定种子的合成任务评估仍不足以证明真实数据上的表现。调参使用验证集，最终评价使用未参与调参的测试集；需要进一步比较多个种子和真实数据。标准化参数只从训练集计算，避免数据泄漏。

回归可报告测试集 MSE，分类可报告准确率与混淆矩阵。聚类可比较目标值、不同初始化和数据分布。CSV、可靠性评估、经典算法扩展和公平模型比较已有独立案例；下一步按[深度学习计划中的补充路线](50_deep_learning_plan.md)补模型与预处理参数持久化。

## CSV、标准化与独立测试评估

`src/ml/23_csv_standardization.c` 从 `data/ml/study_outcomes.csv` 读取两个数值特征、二分类标签和明确的数据划分。运行 `make csv-standardization` 查看训练集拟合的均值/标准差、验证集选择的训练轮次，以及只在训练结束后计算的测试集 BCE、准确率和混淆矩阵。`make test-csv-standardization` 检查 CSV 错误处理、训练集统计量隔离与测试集计数；`make sanitize-csv-standardization` 检查内存和未定义行为。

示例只接受固定表头、四个未加引号字段的教学用 CSV 子集，不是完整 RFC 4180 解析器。数据是手工构造的小型示例，不代表真实考试或任何真实人群的表现。逐函数源码地图、标准化和梯度推导、首条测试样本计算见 [CSV 标准化技术分析](43_csv_standardization_analysis.md)。

## 类别不平衡与概率校准

`src/ml/24_reliability_assessment.c` 用固定种子的合成数据演示正类稀少时准确率为何会误导，并比较多数类基线、普通损失和类别加权损失。运行 `make reliability-assessment` 查看混淆矩阵、精确率、召回率、F1、平衡准确率、验证集选择阈值、Brier 分数、交叉熵、ECE 和 ASCII 可靠性图。阈值及 Platt 映射只使用验证集，测试集只在最后评估。运行 `make test-reliability-assessment` 检查指标边界、确定性与数据隔离；`make sanitize-reliability-assessment` 运行地址和未定义行为消毒器。

可靠性图使用 $10$ 个等宽概率区间，比较每箱平均预测概率与实际正例率。数据集刻意保持正类占比为 $10\%$，每个划分中的正例数较少，因此分箱指标会有明显波动；这是固定教学实验，不是效果保证。完整源码地图、权重/指标/校准推导及测试集实例见 [可靠性评估技术分析](44_reliability_assessment_analysis.md)。

## 扩展经典算法

补充路线第 3 步增加了三种互补算法，均使用固定小数据和纯 C 实现。`make softmax-regression` 运行三分类线性模型，观察稳定 Softmax 与多类交叉熵；`make dbscan` 运行密度聚类，观察核心点扩展、边界点和噪声；`make gradient-boosting` 运行平方误差回归，观察树桩逐轮拟合残差，并用验证集选择轮数。

分别运行 `make test-softmax-regression`、`make test-dbscan` 和 `make test-gradient-boosting` 核对手算结果与边界；对应消毒器目标为 `make sanitize-softmax-regression`、`make sanitize-dbscan` 和 `make sanitize-gradient-boosting`。源码走读见 [Softmax 多分类分析](45_softmax_regression_analysis.md)、[DBSCAN 分析](46_dbscan_analysis.md)和[梯度提升分析](47_gradient_boosting_analysis.md)。这些固定样本上的结果用于验证实现和理解步骤，不作为现实数据上的效果保证。

## 公平模型比较

`src/ml/28_model_comparison.c` 使用同一 CSV 比较多数类基线、逻辑回归、K 近邻和决策树桩。每个固定种子将原训练与验证行合并为开发集，再按类别分层拆分训练/验证；测试行固定不变。每轮只用训练子集拟合标准化参数，所有模型复用同一组统计量和拆分。逻辑回归学习率及 KNN 的邻居数只按验证平衡准确率选择，树桩阈值仅从训练数据学习。

运行 `make model-comparison` 查看五个种子的单轮预测指标和测试集均值 ± 总体标准差；`make test-model-comparison` 验证共同划分、手算指标、树桩切分与测试集隔离，`make sanitize-model-comparison` 检查内存和未定义行为。逐函数源码地图、公式推导和真实样本计算见 [公平模型比较技术分析](48_model_comparison_analysis.md)。测试集只有 $8$ 条且五轮重复使用同一测试行，因此波动反映开发集划分变化，不是置信区间，也不证明真实泛化效果。
