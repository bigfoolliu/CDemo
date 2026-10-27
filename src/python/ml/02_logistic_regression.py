"""用稳定 sigmoid 和交叉熵实现二分类逻辑回归。"""

from math import exp, fabs, log1p


def sigmoid(value: float) -> float:
    """稳定地把实数映射到概率。

    :param value: 线性模型的 logit。
    :return: [0, 1] 范围内的概率。
    """
    if value >= 0.0:
        return 1.0 / (1.0 + exp(-value))
    exponential = exp(value)
    return exponential / (1.0 + exponential)


def main() -> None:
    """训练一个一维二分类逻辑回归模型。

    :return: 无。
    """
    inputs = [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]
    labels = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    weight = 0.0
    bias = 0.0

    for _ in range(1000):
        dw = 0.0
        db = 0.0
        for value, label in zip(inputs, labels):
            error = sigmoid(weight * value + bias) - label
            dw += error * value / len(inputs)
            db += error / len(inputs)
        weight -= 0.1 * dw
        bias -= 0.1 * db

    loss = 0.0
    for value, label in zip(inputs, labels):
        logit = weight * value + bias
        loss += max(logit, 0.0) - label * logit + log1p(exp(-fabs(logit)))
    loss /= len(inputs)
    print("逻辑回归：交叉熵={:.5f}，w={:.4f}，b={:.4f}".format(loss, weight, bias))
    for value in (-0.5, 0.5):
        print("x={:+.1f} 的正类概率={:.4f}".format(value, sigmoid(weight * value + bias)))


if __name__ == "__main__":
    main()
