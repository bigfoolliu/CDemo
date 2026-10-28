"""用标准库实现 D02 的图像增强与同预算消融实验。"""

import csv
from dataclasses import dataclass
import math
import sys

SIDE = 8
PIXELS = SIDE * SIDE
CLASSES = 3
TRAIN_COUNT = 72
EVALUATION_COUNT = 36
BATCH_SIZE = 9
EPOCHS = 120
RUN_COUNT = 5
WEIGHTS = CLASSES * PIXELS
PARAMETERS = WEIGHTS + CLASSES
NOISE_AMPLITUDE = 0.12
LEARNING_RATE = 0.08
MASK = 0xFFFFFFFF

AUGMENTATIONS = ("none", "translation", "noise", "combined", "bad_rotate")
VIEWS = ("clean", "shifted", "noisy", "shifted_noisy")


@dataclass(frozen=True)
class Image:
    """固定大小的灰度图、类别标签和原始样本标识。"""

    pixels: tuple[float, ...]
    target: int
    source_id: int


@dataclass(frozen=True)
class Metrics:
    """一个数据视图上的平均交叉熵和准确率。"""

    loss: float
    accuracy: float


@dataclass(frozen=True)
class RunResult:
    """一个增强设置和训练种子的验证、测试结果。"""

    validation: Metrics
    test: tuple[Metrics, ...]


def random_next(state: int) -> int:
    """按 uint32 线性同余规则生成下一个状态。

    :param state: 上一状态，按 32 位无符号整数解释。
    :return: 下一 32 位状态。
    """
    return (state * 1664525 + 1013904223) & MASK


def random_unit(state: int) -> tuple[int, float]:
    """从状态生成与 C 实现一致的 [0, 1) 数值。

    :param state: 上一伪随机状态。
    :return: 新状态及由高 24 位生成的浮点数。
    """
    state = random_next(state)
    return state, (state >> 8) / 16777216.0


def make_dataset(count: int, seed: int, source_group: int) -> tuple[Image, ...]:
    """独立生成类别均衡的圆形、方形和朝上三角形。

    :param count: 正数且须为类别数的整数倍，最多 72。
    :param seed: 此数据划分专用的确定性种子。
    :param source_group: 训练、验证、测试各自不同的来源组编号。
    :return: 不共享来源编号的固定灰度图元组。
    :raises ValueError: 数量不满足固定容量或类别均衡条件时抛出。
    """
    if count <= 0 or count > TRAIN_COUNT or count % CLASSES != 0:
        raise ValueError("数据数量必须是 3 的正整数倍且不超过固定容量")
    images = []
    for index in range(count):
        state = (seed + (index + 1) * 2654435761) & MASK
        state, random_x = random_unit(state)
        state, random_y = random_unit(state)
        state, random_radius = random_unit(state)
        center_x = 3.5 + 0.5 * (random_x - 0.5)
        center_y = 3.5 + 0.5 * (random_y - 0.5)
        brightness = 0.8 + 0.2 * random_radius
        target = index % CLASSES
        pixels = []
        for row in range(SIDE):
            for column in range(SIDE):
                dx = column - center_x
                dy = row - center_y
                inside = abs(dx) <= 1 and abs(dy) <= 1
                if target == 1:
                    inside = abs(dx) <= 0.5 and abs(dy) <= 1.5
                elif target == 2:
                    inside = abs(dx) <= 1.5 and abs(dy) <= 0.5
                pixels.append(brightness if inside else 0.0)
        images.append(Image(tuple(pixels), target, source_group * 1024 + index))
    return tuple(images)


def translate(pixels: tuple[float, ...], dx: int, dy: int) -> tuple[float, ...] | None:
    """平移像素；若裁掉前景则拒绝并返回无结果。

    :param pixels: 64 个位于 [0, 1] 的行优先像素。
    :param dx: 水平移动量，范围为 -1 到 1。
    :param dy: 垂直移动量，范围为 -1 到 1。
    :return: 新像素元组；超出移动范围或裁掉前景时返回 None。
    """
    if len(pixels) != PIXELS or dx not in (-1, 0, 1) or dy not in (-1, 0, 1):
        return None
    output = [0.0] * PIXELS
    for row in range(SIDE):
        for column in range(SIDE):
            value = pixels[row * SIDE + column]
            target_row = row + dy
            target_column = column + dx
            if value != 0 and not (0 <= target_row < SIDE and 0 <= target_column < SIDE):
                return None
            if 0 <= target_row < SIDE and 0 <= target_column < SIDE:
                output[target_row * SIDE + target_column] = value
    return tuple(output)


def add_noise(pixels: tuple[float, ...], seed: int, amplitude: float) -> tuple[tuple[float, ...], int]:
    """加入可重现的均匀噪声并将强度限制到 [0, 1]。

    :param pixels: 64 个有效灰度值。
    :param seed: 噪声生成器初始状态。
    :param amplitude: 噪声绝对幅度上界，须为有限非负数。
    :return: 加噪像素与最终随机状态。
    :raises ValueError: 像素长度、取值或幅度不合法时抛出。
    """
    if len(pixels) != PIXELS or not math.isfinite(amplitude) or amplitude < 0:
        raise ValueError("噪声参数无效")
    state = seed & MASK
    output = []
    for value in pixels:
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("像素必须是 [0, 1] 内的有限数")
        state, random_value = random_unit(state)
        noise = amplitude * (2 * random_value - 1)
        output.append(min(1.0, max(0.0, value + noise)))
    return tuple(output), state


def augment(image: Image, mode: str, seed: int) -> Image:
    """按指定策略生成训练视图，保留原图和来源编号。

    :param image: 原始图像，目标类别须在 0 到 2 之间。
    :param mode: none、translation、noise、combined 或 bad_rotate。
    :param seed: 平移与噪声使用的确定性随机种子。
    :return: 不修改原图的新图像；错误旋转会故意保留错误标签。
    :raises ValueError: 图像无效、模式未知或安全平移裁掉前景时抛出。
    """
    if len(image.pixels) != PIXELS or image.target not in range(CLASSES):
        raise ValueError("图像形状或标签无效")
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in image.pixels):
        raise ValueError("像素必须是 [0, 1] 内的有限数")
    if mode not in AUGMENTATIONS:
        raise ValueError("未知数据增强模式")
    pixels = image.pixels
    state = seed & MASK
    if mode in ("translation", "combined"):
        state, random_x = random_unit(state)
        state, random_y = random_unit(state)
        moved = translate(pixels, int(random_x * 3) - 1, int(random_y * 3) - 1)
        if moved is None:
            raise ValueError("平移会裁掉图形前景")
        pixels = moved
    elif mode == "bad_rotate" and image.target != 0:
        pixels = tuple(
            image.pixels[(SIDE - 1 - column) * SIDE + row]
            for row in range(SIDE)
            for column in range(SIDE)
        )
    if mode in ("noise", "combined"):
        pixels, _ = add_noise(pixels, state, NOISE_AMPLITUDE)
    return Image(pixels, image.target, image.source_id)


def evaluation_image(image: Image, view: str, index: int) -> Image:
    """构造跨所有模型固定不变的干净或受扰测试视图。

    :param image: 原始留出样本。
    :param view: clean、shifted、noisy 或 shifted_noisy。
    :param index: 测试集位置，用于固定扰动排列。
    :return: 不改变标签和来源编号的评估图像。
    :raises ValueError: 视图未知或平移会裁掉图形时抛出。
    """
    if view not in VIEWS:
        raise ValueError("未知测试视图")
    pixels = image.pixels
    if view in ("shifted", "shifted_noisy"):
        moved = translate(pixels, index % 3 - 1, (index // 3) % 3 - 1)
        if moved is None:
            raise ValueError("测试平移会裁掉图形前景")
        pixels = moved
    if view in ("noisy", "shifted_noisy"):
        pixels, _ = add_noise(pixels, 0xD02A0000 + index, NOISE_AMPLITUDE)
    return Image(pixels, image.target, image.source_id)


def forward(weights: list[float], biases: list[float], image: Image) -> tuple[list[float], float]:
    """计算像素线性分类器的稳定 Softmax 和交叉熵。

    :param weights: 按类别、像素顺序展平的 192 个参数。
    :param biases: 三个类别偏置。
    :param image: 输入图像和目标标签。
    :return: 类别概率与单样本交叉熵。
    :raises ValueError: 参数数量或输入非有限时抛出。
    """
    if len(weights) != WEIGHTS or len(biases) != CLASSES or image.target not in range(CLASSES):
        raise ValueError("模型形状或标签无效")
    logits = []
    for category in range(CLASSES):
        value = biases[category]
        for pixel in range(PIXELS):
            value += weights[category * PIXELS + pixel] * image.pixels[pixel]
        if not math.isfinite(value):
            raise ValueError("logit 非有限")
        logits.append(value)
    maximum = max(logits)
    exponentials = [math.exp(value - maximum) for value in logits]
    total = sum(exponentials)
    probabilities = [value / total for value in exponentials]
    loss = maximum - logits[image.target] + math.log(total)
    return probabilities, loss


def gradient(weights: list[float], biases: list[float], image: Image) -> tuple[list[float], float]:
    """计算单样本 Softmax 交叉熵对权重和偏置的梯度。

    :param weights: 当前展平分类权重。
    :param biases: 当前类别偏置。
    :param image: 当前图像与目标标签。
    :return: 按权重、偏置顺序排列的 195 个梯度及损失。
    """
    probabilities, loss = forward(weights, biases, image)
    result = [0.0] * PARAMETERS
    for category in range(CLASSES):
        delta = probabilities[category] - float(category == image.target)
        result[WEIGHTS + category] = delta
        for pixel in range(PIXELS):
            result[category * PIXELS + pixel] = delta * image.pixels[pixel]
    return result, loss


def initialize(seed: int) -> tuple[list[float], list[float]]:
    """用固定线性同余序列初始化小幅度分类权重。

    :param seed: 初始化随机种子。
    :return: 展平权重与类别偏置。
    """
    state = seed & MASK
    values = []
    for _ in range(PARAMETERS):
        state, random_value = random_unit(state)
        values.append(0.04 * (2 * random_value - 1))
    return values[:WEIGHTS], values[WEIGHTS:]


def evaluate(weights: list[float], biases: list[float], dataset: tuple[Image, ...],
             view: str) -> Metrics:
    """在固定数据视图上累计平均损失和准确率。

    :param weights: 已训练模型权重。
    :param biases: 已训练模型偏置。
    :param dataset: 仅用于评估的数据。
    :param view: 固定的干净或受扰视图。
    :return: 数据集平均交叉熵与准确率。
    """
    total_loss = 0.0
    correct = 0
    for index, source in enumerate(dataset):
        image = evaluation_image(source, view, index)
        probabilities, loss = forward(weights, biases, image)
        total_loss += loss
        correct += int(max(range(CLASSES), key=probabilities.__getitem__) == image.target)
    return Metrics(total_loss / len(dataset), correct / len(dataset))


def train(weights: list[float], biases: list[float], training: tuple[Image, ...],
          mode: str, seed: int) -> None:
    """固定轮数和批次数，用增强视图训练同一个线性分类器。

    :param weights: 可变分类权重，会被原地更新。
    :param biases: 可变类别偏置，会被原地更新。
    :param training: 只用于梯度的训练原图集合。
    :param mode: 本轮比较的增强方式。
    :param seed: 初始化之外的洗牌与增强主种子。
    :return: 无。
    """
    shuffle_state = seed ^ 0x2468ACE0
    for epoch in range(EPOCHS):
        order = list(range(len(training)))
        for remaining in range(len(order), 1, -1):
            shuffle_state, random_value = random_unit(shuffle_state)
            selected = int(random_value * remaining)
            order[remaining - 1], order[selected] = order[selected], order[remaining - 1]
        for start in range(0, len(training), BATCH_SIZE):
            batch = order[start:start + BATCH_SIZE]
            accumulated = [0.0] * PARAMETERS
            for sample_index in batch:
                source = training[sample_index]
                augmentation_seed = (seed ^ ((epoch + 1) * 0x9E3779B9) ^
                                     ((sample_index + 1) * 0x85EBCA6B)) & MASK
                transformed = augment(source, mode, augmentation_seed)
                sample_gradient, _ = gradient(weights, biases, transformed)
                for parameter, value in enumerate(sample_gradient):
                    accumulated[parameter] += value / len(batch)
            for parameter in range(PARAMETERS):
                if parameter < WEIGHTS:
                    weights[parameter] -= LEARNING_RATE * accumulated[parameter]
                else:
                    biases[parameter - WEIGHTS] -= LEARNING_RATE * accumulated[parameter]


def run_one(training: tuple[Image, ...], validation: tuple[Image, ...],
            testing: tuple[Image, ...], mode: str, seed: int) -> RunResult:
    """完成一个同预算实验并返回独立评估结果。

    :param training: 固定训练划分。
    :param validation: 不参与梯度的固定验证划分。
    :param testing: 训练和验证选择之外的固定测试划分。
    :param mode: 单项或组合增强条件。
    :param seed: 与其他条件配对的训练随机种子。
    :return: 验证指标以及四种固定测试视图的指标。
    """
    weights, biases = initialize(seed ^ 0x13579BDF)
    train(weights, biases, training, mode, seed)
    validation_metrics = evaluate(weights, biases, validation, "clean")
    test_metrics = tuple(evaluate(weights, biases, testing, view) for view in VIEWS)
    return RunResult(validation_metrics, test_metrics)


def compute_results() -> tuple[tuple[RunResult, ...], ...]:
    """执行相同划分、更新预算和配对种子的五项消融。

    :return: 依增强模式、训练种子排列的实验结果。
    """
    training = make_dataset(TRAIN_COUNT, 2025, 1)
    validation = make_dataset(EVALUATION_COUNT, 2026, 2)
    testing = make_dataset(EVALUATION_COUNT, 2027, 3)
    return tuple(
        tuple(run_one(training, validation, testing, mode, 42 + run) for run in range(RUN_COUNT))
        for mode in AUGMENTATIONS
    )


def csv_rows(results: tuple[tuple[RunResult, ...], ...]) -> list[dict[str, str]]:
    """把逐种子结果转为便于 C/Python 比对的 CSV 记录。

    :param results: compute_results 返回的完整实验结果。
    :return: 字段次序与 C 程序相同的字典列表。
    """
    rows = []
    for mode_index, mode in enumerate(AUGMENTATIONS):
        for run, result in enumerate(results[mode_index]):
            row = {
                "augmentation": mode,
                "seed": str(42 + run),
                "validation_loss": f"{result.validation.loss:.12f}",
                "validation_accuracy": f"{result.validation.accuracy:.12f}",
            }
            for view_index, view in enumerate(VIEWS):
                row[f"{view}_loss"] = f"{result.test[view_index].loss:.12f}"
                row[f"{view}_accuracy"] = f"{result.test[view_index].accuracy:.12f}"
            rows.append(row)
    return rows


def summary(results: tuple[tuple[RunResult, ...], ...]) -> str:
    """格式化各方法跨训练种子的均值与总体标准差。

    :param results: compute_results 返回的完整实验结果。
    :return: 中文终端摘要。
    """
    lines = [
        f"固定划分：训练/验证/测试={TRAIN_COUNT}/{EVALUATION_COUNT}/{EVALUATION_COUNT}；"
        f"每项 {EPOCHS} 轮、{TRAIN_COUNT // BATCH_SIZE} 批/轮、{RUN_COUNT} 次种子重复。",
        "增强方法              验证损失均值±标准差   干净准确率   平移准确率   噪声准确率   平移+噪声准确率",
    ]
    for mode_index, mode in enumerate(AUGMENTATIONS):
        validation_values = [result.validation.loss for result in results[mode_index]]
        validation_mean, validation_std = mean_and_std(validation_values)
        cells = [f"{mode:<20} {validation_mean:.4f} ± {validation_std:.4f}"]
        for view_index in range(len(VIEWS)):
            accuracies = [result.test[view_index].accuracy for result in results[mode_index]]
            mean, std = mean_and_std(accuracies)
            cells.append(f"{mean:.3f} ± {std:.3f}")
        lines.append("     ".join(cells))
    lines.append("测试视图在所有训练和验证选择之外固定；标准差只反映五个训练种子的波动。")
    return "\n".join(lines)


def mean_and_std(values: list[float]) -> tuple[float, float]:
    """计算有限样本的算术均值与总体标准差。

    :param values: 非空有限数值序列。
    :return: 均值与以序列长度为分母的标准差。
    """
    mean = sum(value / len(values) for value in values)
    variance = sum((value - mean) ** 2 / len(values) for value in values)
    return mean, math.sqrt(variance)


def hand_example() -> str:
    """执行一像素三分类的前向、梯度和一次 SGD 更新。

    :return: 可与 C 程序 ``--hand`` 输出逐项核对的数值。
    """
    grid = tuple(1.0 if index == 3 * SIDE + 3 else 0.0 for index in range(PIXELS))
    moved = translate(grid, 1, 0)
    assert moved is not None
    noisy, _ = add_noise(moved, 1, NOISE_AMPLITUDE)
    patch = [
        " ".join(f"{noisy[row * SIDE + column]:.6f}" for column in range(3, 6))
        for row in range(3, 6)
    ]
    image = Image(grid, 0, 0)
    weights = [0.0] * WEIGHTS
    biases = [0.0] * CLASSES
    weights[3 * SIDE + 3] = 0.2
    weights[PIXELS + 3 * SIDE + 3] = 0
    weights[2 * PIXELS + 3 * SIDE + 3] = -0.2
    probabilities, loss = forward(weights, biases, image)
    gradients, _ = gradient(weights, biases, image)
    active = [gradients[category * PIXELS + 3 * SIDE + 3] for category in range(CLASSES)]
    updated = [weights[category * PIXELS + 3 * SIDE + 3] - LEARNING_RATE * active[category]
               for category in range(CLASSES)]
    return "\n".join((
        "变换手算：像素 (3,3) 右移到 (3,4)，再用种子 1 加噪；局部 3×3：",
        *patch,
        "logits=[0.200000000000, 0.000000000000, -0.200000000000]",
        "probabilities=[{:.12f}, {:.12f}, {:.12f}], loss={:.12f}".format(
            *probabilities, loss),
        "active-pixel gradient=[{:.12f}, {:.12f}, {:.12f}]".format(*active),
        "one SGD step (rate=0.08)=[{:.12f}, {:.12f}, {:.12f}]".format(*updated),
    ))


def main() -> None:
    """运行完整消融，并可输出 C/Python 对照用 CSV。

    :return: 无。
    """
    if sys.argv[1:] == ["--hand"]:
        print(hand_example())
    elif sys.argv[1:] == ["--csv"]:
        results = compute_results()
        rows = csv_rows(results)
        writer = csv.DictWriter(sys.stdout, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    elif not sys.argv[1:]:
        print(summary(compute_results()))
    else:
        raise SystemExit(f"用法：{sys.argv[0]} [--hand | --csv]")


if __name__ == "__main__":
    main()
