"""用固定二类和三类分布演示熵、交叉熵、KL 与梯度。"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Metrics:
    """保存信息量及可选的 logit 梯度。"""

    entropy: float
    cross_entropy: float
    kl: float
    logit_gradient: tuple[float, ...] | None


def validate_distribution(values: tuple[float, ...]) -> None:
    """验证二类或三类概率分布。

    :param values: 概率元组，允许零分量。
    :return: 无。
    :raises ValueError: 长度、范围、有限性或总和错误时抛出。
    """
    if (len(values) not in (2, 3) or
            any(not math.isfinite(value) or not 0 <= value <= 1 for value in values) or
            not math.isclose(sum(values), 1, rel_tol=0, abs_tol=1e-12)):
        raise ValueError("需要总和为 1 的二类或三类有限概率")


def distribution_metrics(target: tuple[float, ...],
                         predicted: tuple[float, ...]) -> Metrics:
    """计算熵、交叉熵、KL 和内点概率的 logit 梯度。

    :param target: 目标概率分布 q。
    :param predicted: 预测概率分布 p，长度必须与 target 相同。
    :return: 信息量；若真实事件概率为零，损失与 KL 为无穷大。
    :raises ValueError: 无效分布或长度不一致时抛出。
    """
    validate_distribution(target)
    validate_distribution(predicted)
    if len(target) != len(predicted):
        raise ValueError("两个分布的类别数必须相同")
    entropy = -sum(q * math.log(q) for q in target if q > 0)
    impossible = any(q > 0 and p == 0 for q, p in zip(target, predicted))
    if impossible:
        return Metrics(entropy, math.inf, math.inf, None)
    cross_entropy = -sum(q * math.log(p) for q, p in zip(target, predicted) if q > 0)
    if not math.isfinite(cross_entropy):
        raise ValueError("交叉熵非有限")
    gradient = (tuple(p - q for q, p in zip(target, predicted))
                if all(p > 0 for p in predicted) else None)
    return Metrics(entropy, cross_entropy, cross_entropy - entropy, gradient)


def gaussian_loss(observation: float, mean: float,
                  variance: float) -> tuple[float, float]:
    """计算固定方差高斯负对数似然和均值梯度。

    :param observation: 有限观测值。
    :param mean: 有限预测均值。
    :param variance: 有限且严格正的已知方差。
    :return: 损失和对均值的导数。
    :raises ValueError: 输入或输出无效时抛出。
    """
    if (not all(map(math.isfinite, (observation, mean, variance))) or variance <= 0):
        raise ValueError("高斯参数须有限且方差为正")
    residual = mean - observation
    loss = 0.5 * math.log(2 * math.pi * variance) + residual ** 2 / (2 * variance)
    gradient = residual / variance
    if not math.isfinite(loss) or not math.isfinite(gradient):
        raise ValueError("结果非有限")
    return loss, gradient


def results() -> dict[str, float]:
    """计算固定情景的全部跨语言对照字段。

    :return: 按展示顺序排列的字段和值。
    """
    values: dict[str, float] = {}
    names = ("correct", "wrong", "uncertain")
    binary_cases = ((0.01, 0.99), (0.99, 0.01), (0.5, 0.5))
    ternary_cases = ((0.05, 0.9, 0.05), (0.9, 0.05, 0.05),
                     (1 / 3, 1 / 3, 1 / 3))
    for name, binary_case, ternary_case in zip(names, binary_cases, ternary_cases):
        binary = distribution_metrics((0, 1), binary_case)
        ternary = distribution_metrics((0, 1, 0), ternary_case)
        assert binary.logit_gradient is not None
        assert ternary.logit_gradient is not None
        values[f"binary_{name}_loss"] = binary.cross_entropy
        values[f"binary_{name}_logit_gradient"] = binary.logit_gradient[1]
        values[f"ternary_{name}_loss"] = ternary.cross_entropy
        values[f"ternary_{name}_target_logit_gradient"] = ternary.logit_gradient[1]
    soft = distribution_metrics((0.5, 0.5), (0.75, 0.25))
    equal = distribution_metrics((0.5, 0.5), (0.5, 0.5))
    gaussian_nll, gaussian_gradient = gaussian_loss(2, 1, 1)
    values["soft_entropy"] = soft.entropy
    values["soft_cross_entropy"] = soft.cross_entropy
    values["soft_kl"] = soft.kl
    values["equal_kl"] = equal.kl
    values["gaussian_nll"] = gaussian_nll
    values["gaussian_mean_gradient"] = gaussian_gradient
    return values


def main() -> None:
    """运行固定损失实验并可选输出 CSV。

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
    for name in ("correct", "wrong", "uncertain"):
        print("{}：二类损失 {:.6f}、目标 logit 梯度 {:.6f}；"
              "三类损失 {:.6f}、目标 logit 梯度 {:.6f}".format(
                  name, values[f"binary_{name}_loss"],
                  values[f"binary_{name}_logit_gradient"],
                  values[f"ternary_{name}_loss"],
                  values[f"ternary_{name}_target_logit_gradient"]))
    print("软目标：熵 {:.6f}、交叉熵 {:.6f}、KL {:.6f}；相同分布 KL {:.6f}".format(
        values["soft_entropy"], values["soft_cross_entropy"],
        values["soft_kl"], values["equal_kl"]))
    print("高斯：负对数似然 {:.6f}，均值梯度 {:.6f}".format(
        values["gaussian_nll"], values["gaussian_mean_gradient"]))


if __name__ == "__main__":
    main()
