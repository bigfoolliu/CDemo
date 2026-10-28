"""用固定小矩阵演示归一化、权重初始化和残差梯度。"""

from dataclasses import dataclass
import math
from typing import Sequence

MAX_ROWS = 4
MAX_FEATURES = 4
MAX_INITIALIZED_WEIGHTS = 12
EPSILON = 0.01
MOMENTUM = 0.9
INPUT = ((1.0, 2.0, 3.0), (2.0, 4.0, 6.0))
GAMMA = (1.2, 0.8, -0.5)
BETA = (0.1, -0.2, 0.3)
UPSTREAM = ((1.0, 0.5, -1.0), (-0.5, 1.0, 2.0))
RESIDUAL_DEPTH = 8


@dataclass(frozen=True)
class LayerNormCache:
    """层归一化反向传播所需的标准化值和逆标准差。"""

    rows: int
    features: int
    normalized: tuple[float, ...]
    inverse_scale: tuple[float, ...]


@dataclass(frozen=True)
class BatchNormState:
    """批归一化推理时使用的运行均值与方差。"""

    features: int
    running_mean: tuple[float, ...]
    running_variance: tuple[float, ...]


@dataclass(frozen=True)
class BatchNormCache:
    """批归一化反向传播缓存，记录前向是否处于训练模式。"""

    rows: int
    features: int
    training: bool
    normalized: tuple[float, ...]
    inverse_scale: tuple[float, ...]


@dataclass(frozen=True)
class RmsNormCache:
    """RMSNorm 反向传播所需的标准化值和逆均方根。"""

    rows: int
    features: int
    normalized: tuple[float, ...]
    inverse_scale: tuple[float, ...]


def _flatten_matrix(matrix: Sequence[Sequence[float]]) -> tuple[int, int, tuple[float, ...]]:
    """校验并展平有界的行优先矩阵。

    :param matrix: 非空、等长且元素有限的二维数值序列。
    :return: 行数、列数和按行展平的不可变数据。
    :raises ValueError: 矩阵形状越界、不规则或含非有限数时抛出。
    """
    rows = len(matrix)
    if rows < 1 or rows > MAX_ROWS:
        raise ValueError("行数必须位于固定缓冲区范围内")
    features = len(matrix[0])
    if features < 1 or features > MAX_FEATURES:
        raise ValueError("特征数必须位于固定缓冲区范围内")
    flattened: list[float] = []
    for row in matrix:
        if len(row) != features:
            raise ValueError("矩阵每行必须具有相同特征数")
        for value in row:
            if not math.isfinite(value):
                raise ValueError("矩阵元素必须为有限数")
            flattened.append(float(value))
    return rows, features, tuple(flattened)


def _validate_vector(values: Sequence[float], length: int, name: str) -> tuple[float, ...]:
    """校验指定长度的一维有限数值序列。

    :param values: 待检查的数值序列。
    :param length: 要求的元素数量。
    :param name: 错误消息中使用的参数名。
    :return: 转换为浮点数的不可变序列。
    :raises ValueError: 长度不匹配或存在非有限数时抛出。
    """
    if len(values) != length or any(not math.isfinite(value) for value in values):
        raise ValueError("{} 必须包含 {} 个有限数".format(name, length))
    return tuple(float(value) for value in values)


def layer_norm_forward(
    matrix: Sequence[Sequence[float]],
    gamma: Sequence[float],
    beta: Sequence[float],
    epsilon: float = EPSILON,
) -> tuple[tuple[tuple[float, ...], ...], LayerNormCache]:
    """按每个样本的特征轴计算 LayerNorm。

    :param matrix: 行优先输入矩阵；每行是一个样本。
    :param gamma: 每个特征的缩放参数。
    :param beta: 每个特征的平移参数。
    :param epsilon: 防止方差为零时除零的有限正数。
    :return: 输出矩阵及供反向传播复用的缓存。
    :raises ValueError: 输入、参数、形状或中间结果无效时抛出。
    """
    rows, features, values = _flatten_matrix(matrix)
    scales = _validate_vector(gamma, features, "gamma")
    shifts = _validate_vector(beta, features, "beta")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon 必须为有限正数")

    normalized: list[float] = []
    inverse_scales: list[float] = []
    output: list[float] = []
    for row_index in range(rows):
        start = row_index * features
        row = values[start:start + features]
        mean = sum(row) / features
        if not math.isfinite(mean):
            raise ValueError("LayerNorm 均值不是有限数")
        variance = sum((value - mean) * (value - mean) for value in row) / features
        if not math.isfinite(variance):
            raise ValueError("LayerNorm 方差不是有限数")
        inverse = 1.0 / math.sqrt(variance + epsilon)
        inverse_scales.append(inverse)
        for feature, value in enumerate(row):
            normalized_value = (value - mean) * inverse
            normalized.append(normalized_value)
            output.append(scales[feature] * normalized_value + shifts[feature])
    if any(not math.isfinite(value) for value in output + normalized + inverse_scales):
        raise ValueError("LayerNorm 输出不是有限数")
    cache = LayerNormCache(rows, features, tuple(normalized), tuple(inverse_scales))
    return _matrix_from_flat(output, rows, features), cache


def layer_norm_backward(
    cache: LayerNormCache,
    gamma: Sequence[float],
    upstream: Sequence[Sequence[float]],
) -> tuple[tuple[tuple[float, ...], ...], tuple[float, ...], tuple[float, ...]]:
    """计算 LayerNorm 对输入、缩放和平移参数的梯度。

    :param cache: 对应前向输入的缓存。
    :param gamma: 每个特征的缩放参数。
    :param upstream: 与前向输出同形状的上游梯度。
    :return: 输入梯度矩阵、gamma 梯度和 beta 梯度。
    :raises ValueError: 缓存、形状或梯度输入无效时抛出。
    """
    rows, features, upstream_values = _flatten_matrix(upstream)
    if rows != cache.rows or features != cache.features:
        raise ValueError("上游梯度形状必须与缓存一致")
    scales = _validate_vector(gamma, features, "gamma")
    if len(cache.normalized) != rows * features or len(cache.inverse_scale) != rows:
        raise ValueError("LayerNorm 缓存形状无效")
    input_gradient = [0.0] * (rows * features)
    gamma_gradient = [0.0] * features
    beta_gradient = [0.0] * features
    for row in range(rows):
        start = row * features
        scaled = [upstream_values[start + j] * scales[j] for j in range(features)]
        normalized = cache.normalized[start:start + features]
        sum_scaled = sum(scaled)
        sum_scaled_normalized = sum(scaled[j] * normalized[j] for j in range(features))
        for feature in range(features):
            index = start + feature
            gamma_gradient[feature] += upstream_values[index] * normalized[feature]
            beta_gradient[feature] += upstream_values[index]
            input_gradient[index] = cache.inverse_scale[row] / features * (
                features * scaled[feature] - sum_scaled
                - normalized[feature] * sum_scaled_normalized
            )
    return (
        _matrix_from_flat(input_gradient, rows, features),
        tuple(gamma_gradient),
        tuple(beta_gradient),
    )


def batch_norm_state_init(features: int) -> BatchNormState:
    """创建批归一化运行统计状态。

    :param features: 特征数，范围为 1..MAX_FEATURES。
    :return: 运行均值为零、运行方差为一的新状态。
    :raises ValueError: 特征数超出固定容量时抛出。
    """
    if features < 1 or features > MAX_FEATURES:
        raise ValueError("特征数超出固定缓冲区范围")
    return BatchNormState(features, (0.0,) * features, (1.0,) * features)


def batch_norm_forward(
    matrix: Sequence[Sequence[float]],
    gamma: Sequence[float],
    beta: Sequence[float],
    state: BatchNormState,
    training: bool,
    epsilon: float = EPSILON,
    momentum: float = MOMENTUM,
) -> tuple[tuple[tuple[float, ...], ...], BatchNormCache, BatchNormState]:
    """执行训练态或推理态 BatchNorm，并返回新的运行状态。

    :param matrix: 行优先输入；训练统计沿样本行计算。
    :param gamma: 每个特征的缩放参数。
    :param beta: 每个特征的平移参数。
    :param state: 与特征数一致的旧运行统计。
    :param training: true 使用本批统计并更新运行状态；false 使用运行统计。
    :param epsilon: 有限正稳定项。
    :param momentum: 旧运行统计的占比，范围为 [0,1)。
    :return: 输出矩阵、反向缓存和新运行状态；推理态返回原状态。
    :raises ValueError: 参数、状态、形状或结果无效时抛出。
    """
    rows, features, values = _flatten_matrix(matrix)
    scales = _validate_vector(gamma, features, "gamma")
    shifts = _validate_vector(beta, features, "beta")
    if (state.features != features or len(state.running_mean) != features
            or len(state.running_variance) != features):
        raise ValueError("BatchNorm 运行状态的特征维度不匹配")
    if any(not math.isfinite(value) for value in state.running_mean + state.running_variance):
        raise ValueError("BatchNorm 运行状态必须有限")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon 必须为有限正数")
    if not math.isfinite(momentum) or momentum < 0 or momentum >= 1:
        raise ValueError("momentum 必须位于 [0,1)")

    means: list[float] = []
    variances: list[float] = []
    if training:
        for feature in range(features):
            column = tuple(values[row * features + feature] for row in range(rows))
            mean = sum(column) / rows
            if not math.isfinite(mean):
                raise ValueError("BatchNorm 均值不是有限数")
            variance = sum((value - mean) * (value - mean) for value in column) / rows
            if not math.isfinite(variance):
                raise ValueError("BatchNorm 方差不是有限数")
            means.append(mean)
            variances.append(variance)
        next_state = BatchNormState(
            features,
            tuple(momentum * old + (1.0 - momentum) * new
                  for old, new in zip(state.running_mean, means)),
            tuple(momentum * old + (1.0 - momentum) * new
                  for old, new in zip(state.running_variance, variances)),
        )
    else:
        means.extend(state.running_mean)
        variances.extend(state.running_variance)
        next_state = state

    normalized: list[float] = [0.0] * (rows * features)
    inverse_scales: list[float] = []
    output: list[float] = [0.0] * (rows * features)
    for feature in range(features):
        variance = variances[feature]
        if variance < 0:
            raise ValueError("BatchNorm 方差不能为负")
        inverse = 1.0 / math.sqrt(variance + epsilon)
        inverse_scales.append(inverse)
        for row in range(rows):
            index = row * features + feature
            normalized[index] = (values[index] - means[feature]) * inverse
            output[index] = scales[feature] * normalized[index] + shifts[feature]
    if any(not math.isfinite(value) for value in (
            output + normalized + inverse_scales + list(next_state.running_mean)
            + list(next_state.running_variance))):
        raise ValueError("BatchNorm 输出不是有限数")
    cache = BatchNormCache(rows, features, training, tuple(normalized), tuple(inverse_scales))
    return _matrix_from_flat(output, rows, features), cache, next_state


def batch_norm_backward(
    cache: BatchNormCache,
    gamma: Sequence[float],
    upstream: Sequence[Sequence[float]],
) -> tuple[tuple[tuple[float, ...], ...], tuple[float, ...], tuple[float, ...]]:
    """计算训练态或推理态 BatchNorm 的输入与缩放梯度。

    :param cache: 对应 BatchNorm 前向模式和输入的缓存。
    :param gamma: 每个特征的缩放参数。
    :param upstream: 与前向输出同形状的上游梯度。
    :return: 输入梯度矩阵、gamma 梯度和 beta 梯度。
    :raises ValueError: 缓存、参数或上游梯度形状无效时抛出。
    """
    rows, features, upstream_values = _flatten_matrix(upstream)
    if rows != cache.rows or features != cache.features:
        raise ValueError("上游梯度形状必须与缓存一致")
    scales = _validate_vector(gamma, features, "gamma")
    input_gradient = [0.0] * (rows * features)
    gamma_gradient = [0.0] * features
    beta_gradient = [0.0] * features
    for feature in range(features):
        scaled = [upstream_values[row * features + feature] * scales[feature]
                  for row in range(rows)]
        normalized = [cache.normalized[row * features + feature] for row in range(rows)]
        sum_scaled = sum(scaled)
        sum_scaled_normalized = sum(scaled[i] * normalized[i] for i in range(rows))
        for row in range(rows):
            index = row * features + feature
            gamma_gradient[feature] += upstream_values[index] * normalized[row]
            beta_gradient[feature] += upstream_values[index]
            if cache.training:
                input_gradient[index] = cache.inverse_scale[feature] / rows * (
                    rows * scaled[row] - sum_scaled
                    - normalized[row] * sum_scaled_normalized
                )
            else:
                input_gradient[index] = cache.inverse_scale[feature] * scaled[row]
    return (
        _matrix_from_flat(input_gradient, rows, features),
        tuple(gamma_gradient),
        tuple(beta_gradient),
    )


def rms_norm_forward(
    matrix: Sequence[Sequence[float]],
    gamma: Sequence[float],
    epsilon: float = EPSILON,
) -> tuple[tuple[tuple[float, ...], ...], RmsNormCache]:
    """按每个样本的特征均方根执行 RMSNorm。

    :param matrix: 行优先有限输入矩阵。
    :param gamma: 每个特征的缩放参数。
    :param epsilon: 有限正稳定项。
    :return: 输出矩阵及反向传播缓存。
    :raises ValueError: 参数、形状或结果无效时抛出。
    """
    rows, features, values = _flatten_matrix(matrix)
    scales = _validate_vector(gamma, features, "gamma")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon 必须为有限正数")
    normalized: list[float] = []
    inverse_scales: list[float] = []
    output: list[float] = []
    for row in range(rows):
        start = row * features
        row_values = values[start:start + features]
        mean_square = sum(value * value for value in row_values) / features
        if not math.isfinite(mean_square):
            raise ValueError("RMSNorm 均方不是有限数")
        inverse = 1.0 / math.sqrt(mean_square + epsilon)
        inverse_scales.append(inverse)
        for feature, value in enumerate(row_values):
            normalized_value = value * inverse
            normalized.append(normalized_value)
            output.append(scales[feature] * normalized_value)
    if any(not math.isfinite(value) for value in output + normalized + inverse_scales):
        raise ValueError("RMSNorm 输出不是有限数")
    cache = RmsNormCache(rows, features, tuple(normalized), tuple(inverse_scales))
    return _matrix_from_flat(output, rows, features), cache


def rms_norm_backward(
    cache: RmsNormCache,
    gamma: Sequence[float],
    upstream: Sequence[Sequence[float]],
) -> tuple[tuple[tuple[float, ...], ...], tuple[float, ...]]:
    """计算 RMSNorm 对输入和缩放参数的梯度。

    :param cache: 对应前向输入的缓存。
    :param gamma: 每个特征的缩放参数。
    :param upstream: 与前向输出同形状的上游梯度。
    :return: 输入梯度矩阵和 gamma 梯度。
    :raises ValueError: 缓存、参数或上游梯度形状无效时抛出。
    """
    rows, features, upstream_values = _flatten_matrix(upstream)
    if rows != cache.rows or features != cache.features:
        raise ValueError("上游梯度形状必须与缓存一致")
    scales = _validate_vector(gamma, features, "gamma")
    input_gradient = [0.0] * (rows * features)
    gamma_gradient = [0.0] * features
    for row in range(rows):
        start = row * features
        normalized = cache.normalized[start:start + features]
        scaled = [upstream_values[start + j] * scales[j] for j in range(features)]
        normalized_dot = sum(scaled[j] * normalized[j] for j in range(features))
        for feature in range(features):
            index = start + feature
            gamma_gradient[feature] += upstream_values[index] * normalized[feature]
            input_gradient[index] = cache.inverse_scale[row] * (
                scaled[feature] - normalized[feature] * normalized_dot / features
            )
    return (
        _matrix_from_flat(input_gradient, rows, features),
        tuple(gamma_gradient),
    )


def initializer_variance(fan_in: int, fan_out: int, kind: str) -> float:
    """计算 He 或 Glorot 均匀初始化的目标权重方差。

    :param fan_in: 输入连接数，必须为正。
    :param fan_out: 输出连接数，必须为正。
    :param kind: ``"he"`` 或 ``"glorot"``。
    :return: He 的 $2/fan_in$ 或 Glorot 的 $2/(fan_in+fan_out)$。
    :raises ValueError: 连接数或初始化名称无效时抛出。
    """
    if fan_in <= 0 or fan_out <= 0 or kind not in ("he", "glorot"):
        raise ValueError("fan-in、fan-out 必须为正且初始化名称有效")
    denominator = fan_in if kind == "he" else fan_in + fan_out
    return 2.0 / denominator


def initialize_weights(
    count: int,
    fan_in: int,
    fan_out: int,
    kind: str,
    seed: int,
) -> tuple[tuple[float, ...], float]:
    """使用与 C 版本相同的固定 LCG 初始化并计算有限样本方差。

    :param count: 权重数量，范围为 1..MAX_INITIALIZED_WEIGHTS。
    :param fan_in: 输入连接数。
    :param fan_out: 输出连接数。
    :param kind: ``"he"`` 或 ``"glorot"``。
    :param seed: 固定整数种子。
    :return: 权重元组及按总体方差计算的样本方差。
    :raises ValueError: 形状或初始化参数无效时抛出。
    """
    if count < 1 or count > MAX_INITIALIZED_WEIGHTS:
        raise ValueError("权重数超出固定缓冲区范围")
    target = initializer_variance(fan_in, fan_out, kind)
    limit = math.sqrt(3.0 * target)
    values: list[float] = []
    state = seed & 0xFFFFFFFF
    for _ in range(count):
        state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
        unit = (state >> 8) / 16777216.0
        values.append(limit * (2.0 * unit - 1.0))
    mean = sum(values) / count
    variance = sum((value - mean) ** 2 for value in values) / count
    return tuple(values), variance


def gradient_chain(depth: int, branch_derivative: float, residual: bool) -> tuple[float, float]:
    """重复计算普通主分支或含恒等捷径的标量梯度。

    :param depth: 层数，范围为 1..1024。
    :param branch_derivative: 每层主分支导数，必须有限。
    :param residual: true 时每层导数为 $1+F'(x)$。
    :return: 输入为 1 时的末层值，以及单位上游梯度传回输入的导数。
    :raises ValueError: 层数、导数或结果无效时抛出。
    """
    if depth < 1 or depth > 1024 or not math.isfinite(branch_derivative):
        raise ValueError("层数或主分支导数无效")
    gain = branch_derivative + (1.0 if residual else 0.0)
    value = 1.0
    gradient = 1.0
    for _ in range(depth):
        value *= gain
        gradient *= gain
        if not math.isfinite(value) or not math.isfinite(gradient):
            raise ValueError("重复梯度链产生非有限数")
    return value, gradient


def _matrix_from_flat(values: Sequence[float], rows: int, features: int) -> tuple[tuple[float, ...], ...]:
    """把行优先数组重建为不可变矩阵。

    :param values: 行优先元素。
    :param rows: 行数。
    :param features: 每行元素数。
    :return: 由行元组组成的矩阵。
    """
    return tuple(tuple(values[row * features:(row + 1) * features]) for row in range(rows))


def _norm(values: Sequence[float]) -> float:
    """计算有限向量的欧几里得范数。

    :param values: 数值序列。
    :return: 向量二范数。
    """
    return math.sqrt(sum(value * value for value in values))


def results() -> dict[str, float]:
    """生成与 C ``--csv`` 相同字段的固定计算结果。

    :return: 字段名到有限演示数值的映射。
    """
    layer_output, layer_cache = layer_norm_forward(INPUT, GAMMA, BETA)
    layer_dx, layer_dgamma, layer_dbeta = layer_norm_backward(
        layer_cache, GAMMA, UPSTREAM)
    state = batch_norm_state_init(len(GAMMA))
    batch_output, batch_cache, state = batch_norm_forward(
        INPUT, GAMMA, BETA, state, True)
    batch_dx, batch_dgamma, batch_dbeta = batch_norm_backward(
        batch_cache, GAMMA, UPSTREAM)
    evaluation_output, _, _ = batch_norm_forward(
        ((3.0, 6.0, 9.0),), GAMMA, BETA, state, False)
    rms_output, rms_cache = rms_norm_forward(INPUT, GAMMA)
    rms_dx, rms_dgamma = rms_norm_backward(rms_cache, GAMMA, UPSTREAM)

    values: dict[str, float] = {}
    arrays = {
        "ln_y": layer_output,
        "ln_dx": layer_dx,
        "bn_y": batch_output,
        "bn_dx": batch_dx,
        "rms_y": rms_output,
        "rms_dx": rms_dx,
    }
    for name, matrix in arrays.items():
        for index, value in enumerate(value for row in matrix for value in row):
            values["{}_{}".format(name, index)] = value
    for feature in range(len(GAMMA)):
        values["ln_dgamma_{}".format(feature)] = layer_dgamma[feature]
        values["ln_dbeta_{}".format(feature)] = layer_dbeta[feature]
        values["bn_dgamma_{}".format(feature)] = batch_dgamma[feature]
        values["bn_dbeta_{}".format(feature)] = batch_dbeta[feature]
        values["rms_dgamma_{}".format(feature)] = rms_dgamma[feature]
        values["bn_running_mean_{}".format(feature)] = state.running_mean[feature]
        values["bn_running_variance_{}".format(feature)] = state.running_variance[feature]
        values["bn_inference_y_{}".format(feature)] = evaluation_output[0][feature]
    values["ln_gradient_norm"] = _norm(value for row in layer_dx for value in row)
    values["bn_gradient_norm"] = _norm(value for row in batch_dx for value in row)
    values["rms_gradient_norm"] = _norm(value for row in rms_dx for value in row)
    values["identity_gradient_norm"] = _norm(value for row in UPSTREAM for value in row)
    values["he_target_variance"] = initializer_variance(4, 3, "he")
    values["glorot_target_variance"] = initializer_variance(4, 3, "glorot")
    _, values["he_sample_variance"] = initialize_weights(
        MAX_INITIALIZED_WEIGHTS, 4, 3, "he", 41)
    values["plain_activation"], values["plain_gradient"] = gradient_chain(
        RESIDUAL_DEPTH, -0.1, False)
    values["residual_activation"], values["residual_gradient"] = gradient_chain(
        RESIDUAL_DEPTH, -0.1, True)
    return values


def _print_demo() -> None:
    """显示固定小批量的归一化输出和梯度传播对比。"""
    values = results()
    print("=== 归一化、初始化与残差梯度实验（固定 2×3 输入）===")
    print("输入 X=[1,2,3; 2,4,6]，gamma=[1.2,0.8,-0.5]，epsilon=0.01")
    for name in ("ln_y", "bn_y", "rms_y"):
        output = [values["{}_{}".format(name, i)] for i in range(3)]
        print("{} 第一行输出：[{}]".format(
            name, ", ".join("{:.6f}".format(value) for value in output)))
    print("输入梯度 L2：恒等基线 {:.6f}，LayerNorm {:.6f}，BatchNorm {:.6f}，RMSNorm {:.6f}".format(
        values["identity_gradient_norm"], values["ln_gradient_norm"],
        values["bn_gradient_norm"], values["rms_gradient_norm"]))
    print("8 层主分支导数 -0.1：普通链梯度 {:.3e}，恒等残差梯度 {:.6f}".format(
        values["plain_gradient"], values["residual_gradient"]))


def main() -> None:
    """运行固定归一化与梯度诊断案例。

    :return: 无。
    """
    for field, value in results().items():
        if not math.isfinite(value):
            raise RuntimeError("结果 {} 不是有限数".format(field))
    _print_demo()


if __name__ == "__main__":
    main()
