"""用固定数据比较 Softmax 稳定性、差分步长和近奇异矩阵。"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

STEPS = (1e-1, 1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14)


@dataclass(frozen=True)
class PerturbationResult:
    """保存近奇异线性系统的两个解及扰动放大量。"""

    base_x: tuple[float, float]
    changed_x: tuple[float, float]
    condition_infinity: float
    relative_amplification: float


def naive_softmax(logits: tuple[float, float, float]) -> tuple[float, ...] | None:
    """直接取指数；溢出时返回 None 供教学对照。

    :param logits: 三个有限类别分数。
    :return: 成功时的概率，溢出时为 None。
    :raises ValueError: 输入不是三个有限数时抛出。
    """
    if len(logits) != 3 or any(not math.isfinite(value) for value in logits):
        raise ValueError("需要三个有限分数")
    try:
        values = tuple(math.exp(value) for value in logits)
    except OverflowError:
        return None
    total = sum(values)
    if not math.isfinite(total) or total <= 0:
        return None
    return tuple(value / total for value in values)


def stable_softmax(logits: tuple[float, float, float]) -> tuple[tuple[float, ...], float]:
    """减最大值后求概率及 log-sum-exp。

    :param logits: 三个有限类别分数。
    :return: 概率元组与对数指数和。
    :raises ValueError: 输入或结果非有限时抛出。
    """
    if len(logits) != 3 or any(not math.isfinite(value) for value in logits):
        raise ValueError("需要三个有限分数")
    maximum = max(logits)
    values = tuple(math.exp(value - maximum) for value in logits)
    total = sum(values)
    logsumexp = maximum + math.log(total)
    probabilities = tuple(value / total for value in values)
    if not math.isfinite(logsumexp) or any(not math.isfinite(p) for p in probabilities):
        raise ValueError("稳定计算仍产生非有限结果")
    return probabilities, logsumexp


def cubic_difference(step: float) -> float:
    """在 x=1 处用中心差分近似 x³ 的导数。

    :param step: 有限正步长。
    :return: 中心差分值。
    :raises ValueError: 步长或结果无效时抛出。
    """
    if not math.isfinite(step) or step <= 0:
        raise ValueError("步长须为有限正数")
    high = 1 + step
    low = 1 - step
    high_value = high * high * high
    low_value = low * low * low
    result = (high_value - low_value) / (2 * step)
    if not math.isfinite(result):
        raise ValueError("差分结果非有限")
    return result


def near_singular_system(delta: float, perturbation: float) -> PerturbationResult:
    """求解近奇异二元系统并计算相对误差放大。

    :param delta: 有限正矩阵扰动，1+delta 必须能区别于 1。
    :param perturbation: 有限正右端扰动，必须改变存储值。
    :return: 原解、新解、无穷范数条件数和实际相对放大。
    :raises ValueError: 输入或中间结果无效时抛出。
    """
    if (not math.isfinite(delta) or not math.isfinite(perturbation) or
            delta <= 0 or perturbation <= 0):
        raise ValueError("扰动必须为有限正数")
    a22 = 1 + delta
    determinant = a22 - 1
    b1 = 2.0
    b2 = 2 + determinant
    changed_b2 = b2 + perturbation
    if determinant <= 0 or changed_b2 == b2 or not math.isfinite(changed_b2):
        raise ValueError("扰动在浮点数中消失或系统不可解")
    base_x2 = (b2 - b1) / determinant
    changed_x2 = (changed_b2 - b1) / determinant
    base_x = (b1 - base_x2, base_x2)
    changed_x = (b1 - changed_x2, changed_x2)
    condition = (2 + determinant) * (2 + determinant) / determinant
    input_relative = abs(changed_b2 - b2) / max(abs(b1), abs(b2))
    output_relative = abs(changed_x2 - base_x2)
    amplification = output_relative / input_relative
    if any(not math.isfinite(value) for value in (*base_x, *changed_x,
                                                   condition, amplification)):
        raise ValueError("计算结果非有限")
    return PerturbationResult(base_x, changed_x, condition, amplification)


def results() -> dict[str, float]:
    """计算与 C 对照的全部固定字段。

    :return: 稳定概率、差分误差及矩阵扰动量。
    """
    probabilities, logsumexp = stable_softmax((1000, 1001, 1002))
    system = near_singular_system(1e-8, 1e-10)
    values = {f"softmax_{i}": p for i, p in enumerate(probabilities)}
    values["logsumexp"] = logsumexp
    for index, step in enumerate(STEPS):
        values[f"error_{index}"] = abs(cubic_difference(step) - 3)
    values["changed_x0"] = system.changed_x[0]
    values["changed_x1"] = system.changed_x[1]
    values["condition_infinity"] = system.condition_infinity
    values["relative_amplification"] = system.relative_amplification
    return values


def main() -> None:
    """输出可阅读的实验表或供跨语言对照的 CSV。

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
    print("大分数稳定 Softmax：({:.8f}, {:.8f}, {:.8f})，log-sum-exp={:.8f}".format(
        values["softmax_0"], values["softmax_1"], values["softmax_2"],
        values["logsumexp"]))
    print("中心差分 f(x)=x^3 在 x=1 的解析导数为 3：")
    for index, step in enumerate(STEPS):
        derivative = cubic_difference(step)
        print(f"h={step:.0e}，差分={derivative:.12g}，绝对误差={values[f'error_{index}']:.6g}")
    system = near_singular_system(1e-8, 1e-10)
    print("近奇异系统：x=({:.8f}, {:.8f}) → ({:.8f}, {:.8f})，"
          "条件数≈{:.3g}，实测相对放大≈{:.3g}".format(
              *system.base_x, *system.changed_x, system.condition_infinity,
              system.relative_amplification))


if __name__ == "__main__":
    main()
