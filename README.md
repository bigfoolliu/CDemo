# 现代 C 语言学习示例

本项目通过短小、可运行的程序学习 C 语言，以最新正式标准 **C23（ISO/IEC 9899:2024）** 为目标。C2y 是下一版标准的工作名称，相关实验特性应单独讨论。

## 快速开始

需要 C 编译器和 Make，无第三方依赖。在仓库根目录运行：

```sh
make           # 编译所有示例
make run       # 按编号运行示例
make test      # 运行示例及其中的行为断言
make lint      # 严格警告与语法检查
make clean     # 删除示例可执行文件
make advanced  # 只构建并运行深入示例
make sanitize  # 对深入示例执行地址与未定义行为检查
make ml        # 构建并运行机器学习案例
make stdlib    # 构建并运行常用标准库示例
```

## 文档阅读顺序

`docs/` 中的文档已按学习路线添加数字前缀。零基础建议从 [文档阅读顺序](docs/00_reading_order.md) 开始；希望系统学习机器学习、深度学习、大模型与 Agent，可按[分阶段学习路线](docs/70_ml_to_agent_roadmap.md)同步完成数学手算、纯 C 案例和验证。

`src/*.c` 和 `src/basics/*.c` 都有独立的 `main`；`src/modules/` 演示多个源码链接成一个程序。构建结果放在 `build/`。可用 `make CC=clang` 或 `make CC=gcc` 指定编译器；切换编译器或参数前先执行 `make clean`。

Makefile 优先选择 `-std=c23`；不支持此名称时使用 `-std=c2x`。接受标准选项并不代表完整实现 C23。示例使用真正的新语法，缺少支持时会编译失败，以便明确识别工具链限制。macOS 上名为 `gcc` 的命令也可能实际是 Apple Clang。

## VS Code 开发

仓库提供扩展推荐、clangd 代码提示、Make 构建任务、CodeLLDB 调试入口及 `.clang-format`。首次打开后安装推荐扩展，按 `Cmd+Shift+B` 构建，在“运行和调试”选择示例并按 F5。

配置原因、环境要求、断点操作及常见问题见 [VS Code 开发指南](docs/02_vscode.md)。也可运行 `make compile-commands` 更新代码提示，运行 `make debug` 构建独立调试版本。

## 教学 CPU 模拟器

运行 `make cpu`，通过十条自定义指令学习取指、解码与执行；两个机器码案例演示累加 1～10 和内存读写，均输出 55。核心使用纯 C，包含八个 16 位寄存器、64 KiB 内存、条件跳转与有界执行。

阅读 [CPU 模拟教程](docs/07_cpu_emulation.md)。`./build/cpu/demo --trace` 将逐指令状态同时输出到终端并追加到 `logs/cpu/trace.log`；`make test-cpu` 和 `make sanitize-cpu` 验证核心，VS Code 选择 `C: 教学 CPU` 调试。该指令集用于教学，不运行真实 x86/RISC-V 程序。

## 常用标准库学习专区

阅读 [C 常用标准库学习指南](docs/03_c_standard_library.md)，按头文件导航、API 契约和小练习逐步学习。`src/stdlib/` 提供 10 个独立示例：输入输出、字符串与字节、动态内存、数值转换、字符分类、类型与范围、排序搜索、数学、时间、可变参数。

运行 `make stdlib` 学习本专区，或在 VS Code 中选择 `C: stdlib/对应文件.c`，设置断点后按 F5。示例包含中文函数及参数说明、正常和边界验证，仅使用 ISO C 接口，尚未覆盖全部标准库 API。

## 哈希表与词频统计器

运行 `make word-frequency` 统计内置英文短文，也可运行 `./build/advanced/07_word_frequency --file README.md --top 15` 读取文件，或用 `--stdin` 从管道读取。结果按词频降序、同频按ASCII字典序输出；只识别ASCII英文字母，其他字节作为分隔符。

可复用的线性探测哈希表位于 `src/common/word_map.h`，程序位于 `src/advanced/07_word_frequency.c`。阅读 [词频统计教程](docs/05_word_frequency.md) 学习哈希碰撞、扩容、键所有权与流式分词；运行 `make test-word-frequency` 和 `make sanitize-word-frequency` 验证边界与失败路径。VS Code选择 `C: 哈希表与词频统计` 按F5。

## 图像滤镜实验室

运行 `make filters`，无需SDL或OpenCV，将合成测试图与均值模糊、Gaussian模糊、锐化、浮雕、Sobel结果导出到 `exports/`，共七张BMP，保留已有文件。默认使用边缘复制；`./build/vision/filters_demo --export --zero` 对照零填充边界。

核心为 `src/vision/image_filters.h/c`，入口为 `filters_demo.c`。阅读 [滤镜学习教程](docs/20_image_filters.md)，学习RGB转灰度、3×3互相关、边界采样与浮点响应；显示时才限幅到0..255。`make test-filters`、`make sanitize-filters` 无需窗口，默认构建/测试/检查包含本例。VS Code选择 `C: 图像滤镜实验室` 按F5。

## 落沙模拟

运行 `make sand`，用 1/2/3 选择沙、水、墙，左键绘制，右键擦除，按住左键持续添加；[ / ] 调整笔刷，Space 暂停，N 单步，R 重置，C 清空，S 保存 BMP，Esc 退出。观察沙堆、水流和沙沉入水中，窗口需 SDL2 与 pkg-config。

阅读 [落沙模拟教程](docs/19_sand.md)，学习网格局部规则、更新顺序、访问标记和物质量守恒。目标显示 60 FPS、每秒模拟 60 步；`make test-sand` 与 `make sanitize-sand` 无需 SDL，VS Code 选择 `C: 落沙模拟` 按 F5。

## 波纹模拟

运行 `make ripple`，点击水面观察波纹传播、叠加、反射与衰减。左键凸起、右键凹陷，Space 暂停，N 单步，R 恢复初始波纹，C 清空，S 保存 BMP，Esc 退出。窗口需 SDL2 与 pkg-config，目标显示 60 FPS，固定每秒模拟 60 步。

阅读 [波纹模拟教程](docs/18_ripple.md)，学习高度场、有限差分、双缓冲、阻尼和坡度着色。`make test-ripple` 与 `make sanitize-ripple` 无需 SDL；VS Code 选择 `C: 波纹模拟` 按 F5。

## 贝塞尔曲线编辑器

运行 `make bezier`，左键拖动四个控制点实时编辑三次曲线。绿色 P0、红色 P3 是端点，蓝色 P1/P2 控制弯曲方向；R 重置，S 保存 BMP，Esc 退出。窗口需 SDL2 与 pkg-config，目标显示 60 FPS。

阅读 [贝塞尔曲线教程](docs/16_bezier.md)，学习 de Casteljau 插值、控制多边形、鼠标坐标映射和折线采样。`make test-bezier` 与 `make sanitize-bezier` 无需 SDL；VS Code 选择 `C: 贝塞尔曲线编辑器` 按 F5。

## 曼德勃罗分形

运行 `make mandelbrot`，通过复数迭代生成彩色分形。左键/滚轮放大，右键缩小，方向键平移，+/- 调整迭代上限，R 重置，S 保存 BMP，Esc 退出。图像只在视图改变时重新计算。

阅读 [曼德勃罗分形教程](docs/21_mandelbrot.md)，学习逃逸时间、复平面坐标、鼠标锚点和着色。`make test-mandelbrot` 与 `make sanitize-mandelbrot` 无需 SDL；VS Code 选择 `C: 曼德勃罗分形` 按 F5。

## 迷宫生成与寻路

运行 `make maze`，观察 DFS 回溯生成，再用 BFS 自动寻找路径。Space 暂停，N 单步，R 同种子重建，G 新迷宫，1/2 在同一张完整地图上比较 BFS/DFS，S 保存 BMP，Esc 退出。

阅读 [迷宫算法教程](docs/14_maze.md)，学习显式栈、FIFO 队列、前驱链和最短路。核心测试 `make test-maze` 与 `make sanitize-maze` 无需 SDL；VS Code 选择 `C: 迷宫生成与寻路` 按 F5。

## 寻路算法学习

运行 `make pathfinding`，在可编辑网格上逐步比较 BFS、Dijkstra 和 A*。1/2/3 切换算法，Space 暂停/继续，N 单步，左键切换墙，右键切换普通/高代价地形，R 重置，S 保存 BMP。阅读 [寻路算法教程](docs/15_pathfinding.md)，理解步数最短与总代价最低的区别。`make test-pathfinding`、`make sanitize-pathfinding` 不依赖 SDL；`make test-pathfinding-window` 使用 SDL dummy 验证交互。VS Code 选择 `C: 寻路算法演示` 按 F5。

## 排序算法学习

运行 `make sorting`，以柱状图逐步观察冒泡、插入、选择和快速排序。主键盘或数字小键盘的 1/2/3/4 切换算法，画布左上角会显示当前算法名称；所有算法使用相同的确定性数据。Space 暂停/继续，N 单步，R 换数据，S 保存 BMP。界面用颜色标记比较元素、快速排序枢轴、数组写入和已排定位置。阅读 [排序算法教程](docs/12_sorting.md)，比较复杂度、移动方式与稳定性。`make test-sorting`、`make sanitize-sorting` 不依赖 SDL；`make test-sorting-window` 使用 SDL dummy 验证交互。VS Code 选择 `C: 排序算法演示` 按 F5。

## BMP 图片导出

运行 `make bmp`，无需 SDL 即可绘制并保存 24 位 BMP 到 `exports/`。二维图形、粒子喷泉、三维粒子玫瑰、生命游戏、迷宫、分形、贝塞尔曲线、波纹和落沙窗口均支持按 S 保存当前绘制帧，文件自动命名且不覆盖旧图片。

阅读 [BMP 导出教程](docs/11_bmp_export.md)，学习小端文件头、BGR 通道、上下行顺序和四字节行填充。`make test-bmp`、`make sanitize-bmp` 验证格式；VS Code 选择 `C: BMP 图片导出` 按 F5。

## 康威生命游戏

使用纯 C 二维数组和双缓冲实现 B3/S23 细胞演化，提供滑翔机、闪烁器、可重复随机图案及暂停鼠标编辑。运行 `make life`；Space 暂停，N 单步，1/2/3 切换预设，R 重置，C 清空，Esc 退出。

阅读 [生命游戏教程](docs/13_life.md)。默认绘制目标 60 FPS、每秒演化 10 代，采用固定死亡边界。`make test-life` 和 `make sanitize-life` 无需 SDL；VS Code 选择 `C: 康威生命游戏` 按 F5 调试。

## 粒子喷泉

在二维引擎上用纯 C 实现固定粒子池、重力、寿命回收和颜色渐暗。运行 `make particles`；Space 暂停/继续，暂停时 N 单步，R 重置，Esc 退出。窗口标题显示存活粒子数。

阅读 [粒子效果教程](docs/17_particles.md)。`make test-particles` 和 `make sanitize-particles` 无需 SDL；窗口与二维演示共用 SDL2 依赖。VS Code 选择 `C: 粒子喷泉演示` 按 F5 调试。

## 三维粒子玫瑰

运行 `make rose`，观察由 16,000 个粒子构成的粉白体积云团缓慢旋转、漂动并柔和发光，画面采用竖屏构图。左右/上下方向键调整观察角度，Space 暂停，暂停时 N 单步，R 重置，S 保存 170×240 BMP，Esc 退出。窗口需 SDL2 与 pkg-config；`make test-rose3d`、`make sanitize-rose3d` 验证纯 C 核心，`make test-rose3d-window` 验证窗口与截图。

阅读 [三维粒子玫瑰教程](docs/22_rose3d.md)，学习体积粒子采样、三维旋转、透视投影和加色光晕；运行 `make rose-build` 只构建，`make rose-debug` 构建调试版。VS Code 选择 `C: 三维粒子玫瑰` 按 F5。

## 最小二维图形引擎

`src/graphics/` 提供纯 C 软件绘制核心和可选的 SDL2 实时窗口，演示像素、直线、矩形、圆、二维变换和动画循环。运行 `make graphics`，Space 暂停，暂停时 N 单步，R 重置，Esc 退出。

阅读 [二维图形原理教程](docs/10_graphics_2d.md)。窗口需 SDL2 与 pkg-config，普通构建及 `make test-graphics` 无需这些依赖。VS Code 选择 `C: 二维图形演示`，设置断点后按 F5。

## 基础语法学习目录

先阅读 [C 语言基础学习手册](docs/01_c_basics.md)，再按下表学习。手册包含概念说明、易错点、练习及尚未提供独立示例的深入主题。

| 示例路径（相对于 `src/`） | 内容 |
| --- | --- |
| `basics/01_types.c` | 类型、声明、初始化、常量、sizeof |
| `basics/02_operators.c` | 算术、逻辑、位运算、转换、短路 |
| `basics/03_control_flow.c` | 分支、循环、break、continue |
| `basics/04_functions.c` | 函数原型、按值传参、递归、回调 |
| `basics/05_arrays_strings.c` | 数组、多维数组、字符串及缓冲区 |
| `basics/06_pointers.c` | 指针与数组、生命周期、const、空指针、二级指针及函数指针 |
| `basics/07_aggregate_types.c` | struct、union、enum、typedef |
| `basics/08_scope_storage.c` | 作用域、生命周期与 static |
| `basics/09_dynamic_memory.c` | 分配、扩容、释放及失败处理 |
| `basics/10_preprocessor.c` | 宏、条件编译与 inline 函数 |
| `basics/11_io_errors.c` | 文本解析、文件读写、错误处理 |
| `basics/12_generic_variadic.c` | 泛型选择与可变参数 |
| `modules/` | 头文件、翻译单元、接口与链接 |

例如，单独编译运行第一个基础示例：

```sh
make build/basics/01_types
./build/basics/01_types
```

## 深入示例

阅读 [深入 C 语言学习说明](docs/04_c_advanced.md)，了解接口契约、边界测试与扩展练习。

| 文件（位于 `src/advanced/`） | 核心问题 |
| --- | --- |
| `01_vector.c` | 动态数组、所有权、扩容及分配失败注入 |
| `02_parser.c` | 严格整数解析、极值与溢出前检查 |
| `03_binary_protocol.c` | 字节序、截断检测、零拷贝视图生命周期 |
| `04_callbacks.c` | 回调与上下文、策略复用、提前终止 |
| `05_atomics.c` | 双线程原子计数与 release/acquire 发布 |
| `06_memory_pool.c` | 固定块内存池、对齐、空闲链表、生命周期与失败安全 |
| `07_word_frequency.c` | 开放寻址哈希表、字符串所有权、流式分词与词频排序 |

运行 `make memory-pool` 观察小对象分配与地址复用，配合 [内存池教程](docs/06_memory_pool.md) 学习。`make test-memory-pool` 和 `make sanitize-memory-pool` 验证边界；VS Code 选择 `C: 固定块内存池` 按 F5 调试。

线程示例采用 POSIX API，需要 macOS/Linux 等支持环境；其他深入示例使用 ISO C 接口。消毒器命令要求编译器支持 `-fsanitize=address,undefined`，生成物放在 `build/sanitize/`。

## 机器学习入门案例

`src/ml/` 提供二十七个纯 C 案例，无第三方依赖；大多数使用内置数据，CSV 学习案例读取仓库中的 `data/ml/study_outcomes.csv`。数学函数通过 `-lm` 链接。运行 `make ml`，配合 [中文机器学习说明](docs/30_machine_learning.md) 阅读算法步骤和练习。

| 文件 | 学习目标 |
| --- | --- |
| `01_linear_regression.c` | 梯度下降拟合直线、均方误差与预测 |
| `02_logistic_regression.c` | 二分类、sigmoid、交叉熵及数值稳定性 |
| `03_knn.c` | 距离排序、邻居投票与平局处理 |
| `04_kmeans.c` | 无监督聚类、中心更新与空簇处理 |
| `05_numeric_foundations.c` | 深度学习阶段 0：行优先矩阵、链式法则与梯度检查 |
| `06_xor_mlp.c` | 阶段 1：单个神经元、XOR 小网络、手写反向传播与 SGD |
| `07_mlp_2d.c` | 阶段 2：圆环分类、ReLU、小批量 SGD/Adam 与独立评估 |
| `08_cnn_shapes.c` | 阶段 3：灰度图形 CNN、卷积/池化反向传播与特征图 |
| `09_denoising_autoencoder.c` | 阶段 4：八维瓶颈、图像重建与独立去噪评估 |
| `10_char_rnn.c` | 阶段 5：字符 RNN、完整 BPTT、共享参数梯度与裁剪 |
| `11_gru_memory.c` | 阶段 6：GRU更新/重置门、记忆比特与同预算 RNN 对比 |
| `12_tiny_transformer.c` | 阶段 7：双头因果注意力、位置表示、LayerNorm与完整反向传播 |
| `13_vae_shapes.c` | VAE选修：重参数化、KL、二维潜在空间与图形生成 |
| `14_gan_2d.c` | GAN选修：八簇点生成、交替优化、非饱和损失与模式覆盖 |
| `15_naive_bayes.c` | 阶段 9.1：多项式朴素贝叶斯、词频、平滑与独立短句评估 |
| `16_decision_tree.c` | 阶段 9.2：基尼增益、二维决策树与决策边界 |
| `17_random_forest.c` | 阶段 9.3：bootstrap、随机特征、投票与袋外评估 |
| `18_pca.c` | 阶段 9.4：协方差、幂迭代、二维投影与重建误差 |
| `19_linear_svm.c` | 阶段 9.5：hinge 损失、L2 正则、软间隔与线性分类对照 |
| `20_cross_validation.c` | 阶段 10.1：分层 K 折、折内标准化、正则选参与独立测试集 |
| `21_regularization.c` | 阶段 10.2：小样本过拟合、L2 目标与验证集选参 |
| `22_resnet.c` | 阶段 11.1：恒等捷径、残差块与 CNN 对比 |
| `23_csv_standardization.c` | CSV 读取、训练集标准化、验证集选轮次与独立测试评估 |
| `24_reliability_assessment.c` | 阶段 10.3/10.4：类别不平衡指标、阈值选择、Platt 概率校准 |
| `25_softmax_regression.c` | 多分类 Softmax、交叉熵梯度与训练/验证/测试评估 |
| `26_dbscan.c` | DBSCAN 核心点、边界点、密度扩展与噪声识别 |
| `27_gradient_boosting.c` | 平方损失梯度提升、回归树桩、残差和验证集选轮次 |

运行 `make vae` 导出重建对照、二维潜在网格、插值与先验随机样本四张 BMP；阅读 [VAE 教程](docs/59_vae.md)。`make test-vae` 和 `make sanitize-vae` 验证全部参数及重参数化导数；`--log` 追加日志至 `logs/ml/13_vae_shapes.log`。

运行 `make gan` 导出训练前后散点、判别器热图与潜在网格四张 BMP；阅读 [GAN 教程](docs/60_gan_2d.md)。`make test-gan` 和 `make sanitize-gan` 验证 D→G 梯度链及交替冻结；`--log` 追加至 `logs/ml/14_gan_2d.log`。

运行 `make naive-bayes` 观察普通消息与促销消息的分类、对数分数及多数类基线；阅读 [朴素贝叶斯教程](docs/36_naive_bayes.md)。`make test-naive-bayes` 和 `make sanitize-naive-bayes` 验证概率计算与边界。

运行 `make decision-tree` 比较树桩与深度 2 决策树并导出左右对照 BMP；阅读 [决策树教程](docs/37_decision_tree.md)。`make test-decision-tree` 和 `make sanitize-decision-tree` 验证分裂与边界；直接运行 `./build/ml/16_decision_tree` 不写图片。

运行 `make random-forest` 比较单树与 31 棵树的森林，观察三个种子的测试错误和袋外评估；阅读 [随机森林教程](docs/38_random_forest.md)。`make sanitize-random-forest` 使用地址与未定义行为消毒器运行案例。

运行 `make pca` 用训练集拟合两个主成分，并将训练/测试样本的二维投影图导出到 `exports/`；直接运行 `./build/ml/18_pca` 只显示数值，不写图片。阅读 [PCA 教程](docs/39_pca.md)，学习训练集中心化、协方差、幂迭代及独立测试集重建误差；`make sanitize-pca` 可运行消毒器版本。

运行 `make linear-svm` 训练带 L2 正则的线性 SVM，并与逻辑回归比较训练/测试错误和分类边界；图片保存到 `exports/`。直接运行 `./build/ml/19_linear_svm` 只显示数值；阅读 [线性 SVM 教程](docs/40_linear_svm.md)，学习 hinge 损失、次梯度和几何间隔。`make sanitize-linear-svm` 可运行消毒器版本。

运行 `make cross-validation` 在训练开发集上做 4 折分层交叉验证，按平均 log-loss 选择逻辑回归的 L2 系数；每折只用折内训练部分拟合标准化参数，选参后再用完整开发集重训并评估封存的最终测试集。阅读 [交叉验证教程](docs/41_cross_validation.md)，学习折分、预处理和测试集隔离；`make sanitize-cross-validation` 可运行消毒器版本。

运行 `make csv-standardization` 从 `data/ml/study_outcomes.csv` 加载小型二分类数据，以训练集拟合均值和标准差，用验证集选择训练轮次，再独立评估测试集。阅读 [CSV 标准化技术分析](docs/43_csv_standardization_analysis.md)；`make test-csv-standardization` 验证解析失败安全、训练统计量隔离和测试集评估，`make sanitize-csv-standardization` 运行消毒器版本。

运行 `make reliability-assessment` 学习不平衡分类的准确率陷阱、类别加权、验证集阈值选择，以及验证集拟合的 Platt 概率校准。终端会输出混淆矩阵、精确率、召回率、F1、平衡准确率、Brier、BCE、ECE 和 ASCII 可靠性图。阅读 [可靠性评估技术分析](docs/44_reliability_assessment_analysis.md)；运行 `make test-reliability-assessment` 和 `make sanitize-reliability-assessment` 验证实现。

运行 `make softmax-regression` 学习多分类概率和交叉熵梯度，阅读 [Softmax 分析](docs/45_softmax_regression_analysis.md)；运行 `make test-softmax-regression` 与 `make sanitize-softmax-regression` 验证稳定 Softmax 和手算更新。

运行 `make dbscan` 观察密度聚类如何扩展核心点、重新归类边界点并保留噪声，阅读 [DBSCAN 分析](docs/46_dbscan_analysis.md)；使用 `make test-dbscan` 和 `make sanitize-dbscan` 验证半径边界与失败路径。

运行 `make gradient-boosting` 观察回归树桩逐轮拟合残差，并由验证集选择轮数；阅读 [梯度提升分析](docs/47_gradient_boosting_analysis.md)。`make test-gradient-boosting` 检查手算切分、测试集隔离，`make sanitize-gradient-boosting` 检查内存与未定义行为。

运行 `make model-comparison` 在相同 CSV、分层训练/验证划分和训练集标准化下比较多数类基线、逻辑回归、K 近邻与决策树桩。程序用五个固定种子保留单轮指标并汇总测试集均值与波动；选参只看验证集。阅读 [公平模型比较分析](docs/48_model_comparison_analysis.md)，运行 `make test-model-comparison` 和 `make sanitize-model-comparison` 验证数据隔离、指标与边界。

运行 `make regularization` 比较 24 条含噪样本训练出的 MLP 在不同 L2 强度下的训练/验证损失、权重范数和多种子独立测试结果；系数只按验证集选择。阅读 [正则化教程](docs/42_regularization.md)，学习目标函数、权重梯度与过拟合；`make test-regularization` 检查全部参数梯度，`make sanitize-regularization` 检查内存和未定义行为。

运行 `make resnet` 对照相同参数量和训练预算下的普通 CNN 与 ResNet；阅读 [残差网络教程](docs/61_resnet.md)，理解恒等捷径及其反向梯度。`make test-resnet` 检查两种结构全部参数和输入梯度，并手算恒等路径；`make sanitize-resnet` 检查内存和未定义行为。

这些案例验证合成数据上的算法行为；真实模型评估需要另外划分训练、验证和测试数据。

后续按 [纯 C 深度学习学习计划](docs/50_deep_learning_plan.md) 逐项学习。阶段 0～7 已实现：运行 `make numeric-foundations` 配合 [数值基础教程](docs/51_numeric_foundations.md)，运行 `make xor-mlp` 配合 [XOR 网络教程](docs/52_xor_mlp.md)，运行 `make mlp-2d` 配合 [二维分类教程](docs/53_mlp_2d.md)，运行 `make cnn-shapes` 配合 [CNN 图形教程](docs/54_cnn_shapes.md)，运行 `make autoencoder` 配合 [去噪自编码器教程](docs/55_denoising_autoencoder.md)。图像案例显式运行时将图片导出至 `exports/`；默认运行不生成图片。运行 `make char-rnn` 配合 [字符 RNN 教程](docs/56_char_rnn.md)，观察周期预测与独立语料评估；`./build/ml/10_char_rnn --log` 追加双输出日志至 `logs/ml/10_char_rnn.log`。运行 `make gru-memory` 配合 [GRU 记忆比特教程](docs/57_gru_memory.md)，比较两种网络在不同间隔和种子下的记忆表现。运行 `make tiny-transformer` 配合 [小型 Transformer 教程](docs/58_tiny_transformer.md)，观察字符预测和注意力矩阵；可用 `--log` 追加至 `logs/ml/12_tiny_transformer.log`。之后回看已有 DQN（LSTM 选做）；阶段 9.1～9.5、阶段 10.1～10.4、阶段 11.1 及补充路线第 2～4 步均已实现，第 1 步尚待模型与预处理参数持久化。模型计算不依赖外部框架。

线性回归的详细日志同时输出到屏幕和 `logs/linear_regression.log`，每行包含毫秒时间戳和调用源码位置，多次运行会追加。用 `less logs/linear_regression.log` 分页查看，或运行 `./build/ml/01_linear_regression logs/training.log` 指定日志文件，父目录自动创建。

## C23 新特性示例

| 文件 | 学习内容 | 建议实验 |
| --- | --- | --- |
| `src/01_bool_nullptr.c` | `bool`、`true`、`false`、`nullptr`、`static_assert` | 改变布尔状态，观察断言失败 |
| `src/02_literals.c` | 二进制字面量与数字分隔符 | 改变二进制位并计算十进制结果 |
| `src/03_bitint.c` | `_BitInt(N)` 与无符号整数回绕 | 修改位宽及初始值，预测输出 |

阅读顺序：先看注释，再预测输出，最后编译运行。测试依赖 `assert`，验证时不要设置 `-DNDEBUG`。

## 深度强化学习自动玩贪吃蛇

纯 C 手写的 `11 → 64 → 64 → 3` 神经网络使用 DQN、经验回放和 Adam，训练时复用人工游戏规则。详见 [实现讲解与运行步骤](docs/62_snake_ai.md)。

```sh
make snake-ai
./build/games/snake_ai train --seconds 180 --seed 42
# 将实际输出的运行目录替换到模型路径中：
./build/games/snake_ai eval --model sessions/snake-ai/run-XXXXXX/best.model
./build/games/snake_ai play --model sessions/snake-ai/run-XXXXXX/best.model
```

实验模型、配置和 CSV 指标放在独立的 `sessions/snake-ai/` 目录，训练日志和文本报告放在同编号的 `logs/snake-ai/` 目录；AI 得分与人工历史分开。训练追求分数提升，不保证每局满分。`make test-snake-ai` 验证数值计算与终端自动游戏。

## 终端小游戏

运行 `make snake` 玩实时贪吃蛇：方向键或 WASD 转向，无需回车，蛇持续自动移动；撞墙或撞到自身后一秒自动重开。界面显示当前分数、历史最高及最近五局，历史追加保存至 `snake_scores.log`。Q 或 Ctrl+C 退出。源码位于 `src/games/snake.c`，详见 [游戏说明](docs/08_snake.md)。适用于 macOS/Linux 交互终端，普通 `make run` 不启动游戏；`make test` 验证规则，`make test-snake-terminal` 执行伪终端集成测试（需要 Python 3）。

## 后续学习路线

1. 完成基础示例及手册练习，深入理解对象生命周期与未定义行为。
2. C23 语言特性：`constexpr`、`typeof`、`auto` 类型推导、标准属性、`#embed`。
3. C23 标准库：`<stdckdint.h>` 检查整数溢出、`<stdbit.h>` 位操作。标准库支持需要单独验证。
4. 综合练习：文件处理、错误处理和多文件组织，并使用消毒器检查内存问题。

## 参考资料

- [ISO C23 标准信息](https://www.iso.org/standard/82075.html)
- [WG14 标准委员会与公开文档](https://www.open-std.org/jtc1/sc22/wg14/)
- [Clang C 语言特性支持状态](https://clang.llvm.org/c_status.html)
