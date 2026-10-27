"""用批量梯度下降拟合一元线性回归。"""


def predict(weight: float, bias: float, value: float) -> float:
    """计算一元线性模型的预测值。

    :param weight: 输入的线性权重。
    :param bias: 线性偏置。
    :param value: 待预测的输入值。
    :return: weight * value + bias。
    """
    return weight * value + bias


def main() -> None:
    """用固定数据学习 y = 2x + 1。

    :return: 无。
    """
    inputs = [-2.0, -1.0, 0.0, 1.0, 2.0]
    targets = [-3.0, -1.0, 1.0, 3.0, 5.0]
    weight = 0.0
    bias = 0.0
    rate = 0.05

    for epoch in range(500):
        weight_gradient = 0.0
        bias_gradient = 0.0
        for value, target in zip(inputs, targets):
            error = predict(weight, bias, value) - target
            weight_gradient += 2.0 * error * value / len(inputs)
            bias_gradient += 2.0 * error / len(inputs)
        weight -= rate * weight_gradient
        bias -= rate * bias_gradient

    loss = sum((predict(weight, bias, x) - y) ** 2 for x, y in zip(inputs, targets)) / len(inputs)
    print("线性回归：w={:.4f}，b={:.4f}，MSE={:.8f}".format(weight, bias, loss))
    print("新输入 x=3 的预测：{:.4f}（目标 7）".format(predict(weight, bias, 3.0)))


if __name__ == "__main__":
    main()
