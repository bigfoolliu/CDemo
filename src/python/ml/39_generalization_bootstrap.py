"""通过重复多项式拟合与成对自助法观察泛化误差的不确定性。"""

import argparse
import math

MAX_DEGREE = 5
MAX_SAMPLE_COUNT = 16
DEGREES = (1, 3, 5)
REPEAT_COUNT = 40
PROBE_X = (-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0)
BOOTSTRAP_COUNT = 1000


class Random32:
    """保存与 C 示例相同的 xorshift32 伪随机状态。"""

    def __init__(self, seed: int) -> None:
        """初始化非零随机状态。

        :param seed: 非零的 32 位种子。
        :return: 无。
        :raises ValueError: 种子为零或超出无符号 32 位范围时抛出。
        """
        if seed <= 0 or seed > 0xFFFFFFFF:
            raise ValueError("种子必须是非零 32 位无符号整数")
        self.state = seed

    def next(self) -> int:
        """推进 xorshift32 状态并返回下一个整数。

        :return: 范围为 1 到 2^32-1 的伪随机整数。
        """
        value = self.state
        value ^= (value << 13) & 0xFFFFFFFF
        value ^= value >> 17
        value ^= (value << 5) & 0xFFFFFFFF
        self.state = value & 0xFFFFFFFF
        return self.state

    def unit(self) -> float:
        """将随机整数映射到半开区间 [0, 1)。

        :return: 伪随机浮点数；本生成器仅用于教学。
        """
        return self.next() / 4294967296.0


def target_function(x: float) -> float:
    """计算用于生成数据的无噪声二次函数。

    :param x: 有限输入坐标。
    :return: 函数值 0.75 + 1.2x - 0.8x^2。
    """
    return 0.75 + 1.2 * x - 0.8 * x * x


def dataset_generate(rng: Random32, count: int, lower_x: float, upper_x: float,
                     noise_amplitude: float) -> list[tuple[float, float]]:
    """按给定分布生成带均匀噪声的输入与目标样本。

    :param rng: 调用方维护的确定性随机数生成器。
    :param count: 样本数，范围为 1 到 MAX_SAMPLE_COUNT。
    :param lower_x: 输入区间下界。
    :param upper_x: 大于 lower_x 的输入区间上界。
    :param noise_amplitude: 有限非负的均匀噪声半宽。
    :return: 由 (x, y) 组成的样本列表。
    :raises ValueError: 参数无效或生成非有限样本时抛出。
    """
    values = (lower_x, upper_x, noise_amplitude)
    if (not 1 <= count <= MAX_SAMPLE_COUNT or
            any(not math.isfinite(value) for value in values) or
            lower_x >= upper_x or noise_amplitude < 0):
        raise ValueError("数据生成参数无效")
    samples = []
    for _ in range(count):
        x = lower_x + (upper_x - lower_x) * rng.unit()
        noise = noise_amplitude * (2.0 * rng.unit() - 1.0)
        y = target_function(x) + noise
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("生成的样本不是有限数")
        samples.append((x, y))
    return samples


def polynomial_fit(samples: list[tuple[float, float]], degree: int
                   ) -> tuple[float, ...]:
    """用正规方程和部分主元消元求多项式最小二乘参数。

    :param samples: 至少 degree+1 个有限训练样本，不含验证或测试数据。
    :param degree: 多项式最高次数，范围为 0 到 MAX_DEGREE。
    :return: 按常数项到最高次项排列的参数。
    :raises ValueError: 输入形状无效、设计矩阵秩亏或解非有限时抛出。
    """
    if (not 0 <= degree <= MAX_DEGREE or len(samples) < degree + 1 or
            len(samples) > MAX_SAMPLE_COUNT):
        raise ValueError("多项式拟合形状或次数无效")
    if any(not math.isfinite(x) or not math.isfinite(y) for x, y in samples):
        raise ValueError("训练样本必须为有限数")

    parameter_count = degree + 1
    augmented = [[0.0] * (parameter_count + 1) for _ in range(parameter_count)]
    for row in range(parameter_count):
        for x, y in samples:
            power = 1.0
            for _ in range(row):
                power *= x
            augmented[row][parameter_count] += power * y
            for column in range(parameter_count):
                augmented[row][column] += power
                power *= x

    for column in range(parameter_count):
        pivot = max(range(column, parameter_count),
                    key=lambda row: abs(augmented[row][column]))
        if (not math.isfinite(augmented[pivot][column]) or
                abs(augmented[pivot][column]) < 1e-12):
            raise ValueError("正规方程矩阵秩亏或病态")
        if pivot != column:
            augmented[column][column:], augmented[pivot][column:] = (
                augmented[pivot][column:], augmented[column][column:])
        for row in range(column + 1, parameter_count):
            factor = augmented[row][column] / augmented[column][column]
            for entry in range(column, parameter_count + 1):
                augmented[row][entry] -= factor * augmented[column][entry]

    coefficients = [0.0] * parameter_count
    for row in range(parameter_count - 1, -1, -1):
        value = augmented[row][parameter_count]
        for column in range(row + 1, parameter_count):
            value -= augmented[row][column] * coefficients[column]
        coefficients[row] = value / augmented[row][row]
        if not math.isfinite(coefficients[row]):
            raise ValueError("拟合参数不是有限数")
    return tuple(coefficients)


def polynomial_predict(coefficients: tuple[float, ...], x: float) -> float:
    """使用 Horner 形式计算多项式预测。

    :param coefficients: 按常数项到最高次项排列的有限系数。
    :param x: 有限输入坐标。
    :return: 有限预测值。
    :raises ValueError: 系数为空、次数过高或输入/结果非有限时抛出。
    """
    if (not coefficients or len(coefficients) > MAX_DEGREE + 1 or
            not math.isfinite(x) or any(not math.isfinite(value) for value in coefficients)):
        raise ValueError("预测输入无效")
    prediction = 0.0
    for coefficient in reversed(coefficients):
        prediction = prediction * x + coefficient
    if not math.isfinite(prediction):
        raise ValueError("预测结果不是有限数")
    return prediction


def mean_squared_error(coefficients: tuple[float, ...],
                       samples: list[tuple[float, float]]) -> float:
    """计算模型在独立样本上的均方误差。

    :param coefficients: 已拟合多项式的有限系数。
    :param samples: 非空评估样本，不传入拟合函数。
    :return: 平方误差的算术平均。
    :raises ValueError: 样本为空、含非有限值或结果非有限时抛出。
    """
    if not samples or len(samples) > MAX_SAMPLE_COUNT:
        raise ValueError("评估样本数无效")
    total = 0.0
    for x, y in samples:
        if not math.isfinite(y):
            raise ValueError("评估标签必须为有限数")
        error = polynomial_predict(coefficients, x) - y
        total += error * error
    result = total / len(samples)
    if not math.isfinite(result):
        raise ValueError("均方误差不是有限数")
    return result


def paired_resampled_difference(loss_a: list[float], loss_b: list[float],
                                indices: list[int]) -> float:
    """使用一组共同索引计算两模型的成对平均损失差。

    :param loss_a: 模型 A 在测试样本上的逐样本损失。
    :param loss_b: 模型 B 在相同样本上的逐样本损失。
    :param indices: 有放回抽取的测试样本下标。
    :return: 抽样平均值 mean(loss_A-loss_B)。
    :raises ValueError: 列表长度、索引或损失值无效时抛出。
    """
    if (not loss_a or len(loss_a) != len(loss_b) or len(loss_a) > MAX_SAMPLE_COUNT or
            not indices or len(indices) > MAX_SAMPLE_COUNT or
            any(not math.isfinite(value) for value in (*loss_a, *loss_b)) or
            any(index < 0 or index >= len(loss_a) for index in indices)):
        raise ValueError("成对重采样输入无效")
    return sum(loss_a[index] - loss_b[index] for index in indices) / len(indices)


def paired_bootstrap(loss_a: list[float], loss_b: list[float], replicate_count: int,
                     seed: int) -> tuple[float, float, float]:
    """计算测试集均值差和百分位成对自助置信区间。

    :param loss_a: 模型 A 的逐样本损失。
    :param loss_b: 模型 B 在同一测试样本上的逐样本损失。
    :param replicate_count: 重采样次数，范围为 1 到 BOOTSTRAP_COUNT。
    :param seed: 非零 32 位确定性种子。
    :return: 原测试集均值差、95% 区间下界与上界。
    :raises ValueError: 数组、重采样次数、种子或计算结果无效时抛出。
    """
    if (not loss_a or len(loss_a) != len(loss_b) or len(loss_a) > MAX_SAMPLE_COUNT or
            any(not math.isfinite(value) for value in (*loss_a, *loss_b)) or
            not 1 <= replicate_count <= BOOTSTRAP_COUNT or
            not 1 <= seed <= 0xFFFFFFFF):
        raise ValueError("成对 bootstrap 输入无效")
    rng = Random32(seed)
    mean_difference = sum(a - b for a, b in zip(loss_a, loss_b)) / len(loss_a)
    estimates = []
    for _ in range(replicate_count):
        indices = [rng.next() % len(loss_a) for _ in range(len(loss_a))]
        estimates.append(paired_resampled_difference(loss_a, loss_b, indices))
    estimates.sort()
    lower_index = int(0.025 * (replicate_count - 1))
    upper_index = int(0.975 * (replicate_count - 1))
    interval = (mean_difference, estimates[lower_index], estimates[upper_index])
    if any(not math.isfinite(value) for value in interval):
        raise ValueError("自助法区间不是有限数")
    return interval


def results() -> dict[str, float]:
    """运行重复拟合、固定分布评估、偏移评估和成对自助法。

    :return: 与 C 程序字段逐一对应的确定性统计量。
    """
    prediction_sum = [[0.0] * len(PROBE_X) for _ in DEGREES]
    prediction_square_sum = [[0.0] * len(PROBE_X) for _ in DEGREES]
    rng = Random32(0x1234ABCD)
    for _ in range(REPEAT_COUNT):
        training = dataset_generate(rng, 12, -1.0, 1.0, 0.2)
        for kind, degree in enumerate(DEGREES):
            coefficients = polynomial_fit(training, degree)
            for point, x in enumerate(PROBE_X):
                prediction = polynomial_predict(coefficients, x)
                prediction_sum[kind][point] += prediction
                prediction_square_sum[kind][point] += prediction * prediction

    values: dict[str, float] = {}
    for kind, degree in enumerate(DEGREES):
        bias_squared = 0.0
        variance = 0.0
        for point, x in enumerate(PROBE_X):
            mean = prediction_sum[kind][point] / REPEAT_COUNT
            second_moment = prediction_square_sum[kind][point] / REPEAT_COUNT
            bias_squared += (mean - target_function(x)) ** 2 / len(PROBE_X)
            variance += max(0.0, second_moment - mean * mean) / len(PROBE_X)
        values[f"degree{degree}_bias_squared"] = bias_squared
        values[f"degree{degree}_variance"] = variance
        values[f"degree{degree}_expected_mse"] = bias_squared + variance

    training = dataset_generate(Random32(0x10203041), 12, -1.0, 1.0, 0.2)
    test = dataset_generate(Random32(0x50607081), 12, -1.0, 1.0, 0.2)
    shifted = dataset_generate(Random32(0x90ABCDEF), 12, 1.0, 1.5, 0.2)
    losses_a = []
    losses_b = []
    for kind, degree in enumerate(DEGREES):
        coefficients = polynomial_fit(training, degree)
        values[f"degree{degree}_test_mse"] = mean_squared_error(coefficients, test)
        values[f"degree{degree}_shifted_mse"] = mean_squared_error(coefficients, shifted)
        if kind in (0, len(DEGREES) - 1):
            losses = [((polynomial_predict(coefficients, x) - y) ** 2) for x, y in test]
            if kind == 0:
                losses_a = losses
            else:
                losses_b = losses
    mean_difference, lower, upper = paired_bootstrap(
        losses_a, losses_b, BOOTSTRAP_COUNT, 20261028)
    values["paired_bootstrap_mean"] = mean_difference
    values["paired_bootstrap_lower"] = lower
    values["paired_bootstrap_upper"] = upper
    return values


def main() -> None:
    """打印教学摘要或供跨语言比较的 CSV 数值。

    :return: 无。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", action="store_true", help="输出 field,value")
    args = parser.parse_args()
    values = results()
    if args.csv:
        print("field,value")
        for field, value in values.items():
            print(f"{field},{value:.17g}")
        return

    print(f"多项式模型：每次 12 条训练样本，重复拟合 {REPEAT_COUNT} 次；次数预先固定为 1、3、5。")
    print("次数  平均偏差平方  平均方差  偏差平方+方差  固定测试 MSE  偏移测试 MSE")
    for degree in DEGREES:
        print("{:2d}    {:.6f}      {:.6f}    {:.6f}        {:.6f}       {:.6f}".format(
            degree, values[f"degree{degree}_bias_squared"],
            values[f"degree{degree}_variance"], values[f"degree{degree}_expected_mse"],
            values[f"degree{degree}_test_mse"], values[f"degree{degree}_shifted_mse"]))
    print("测试集只用于固定模型的最终评估；偏移测试把 x 从训练区间 [-1,1] 移到 [1,1.5]。")
    print("成对比较（次数 1 - 次数 5 的逐样本平方损失）：均值={:.6f}，"
          "95% 百分位自助区间=[{:.6f}, {:.6f}]".format(
              values["paired_bootstrap_mean"], values["paired_bootstrap_lower"],
              values["paired_bootstrap_upper"]))
    print("区间以已拟合模型为条件，只反映有限测试样本不确定性；重复训练波动另由偏差方差实验观察。")
    print("小样本合成实验用于学习计算流程，不证明真实任务上的模型优劣。")


if __name__ == "__main__":
    main()
