"""比较不同 L2 强度下逻辑回归参数范数与训练误差。"""

from math import exp


def main() -> None:
    """在小型数据上训练多个正则强度的线性分类器。

    :return: 无。
    """
    samples = [(-2.0, 0), (-1.5, 0), (-1.0, 0), (-0.5, 0),
               (0.5, 1), (1.0, 1), (1.5, 1), (2.0, 1)]
    for penalty in (0.0, 0.1, 1.0):
        weight = 0.0
        for _ in range(400):
            gradient = penalty * weight
            for x, label in samples:
                probability = 1.0 / (1.0 + exp(-max(-30.0, min(30.0, weight * x))))
                gradient += (probability - label) * x / len(samples)
            weight -= 0.1 * gradient
        mistakes = sum((weight * x >= 0.0) != bool(label) for x, label in samples)
        print("lambda={:.1f} weight={:+.3f} |w|={:.3f} 错分数={}".format(
            penalty, weight, abs(weight), mistakes))
    print("正则项惩罚过大的参数；强度应通过验证集选择，不能看测试集调参。")


if __name__ == "__main__":
    main()
