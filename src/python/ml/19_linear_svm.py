"""用次梯度下降训练带 L2 正则的线性软间隔 SVM。"""


def main() -> None:
    """在固定二维数据上优化 hinge 损失并打印分类间隔。

    :return: 无。
    """
    samples = [((-2.0, -1.0), -1), ((-1.0, -2.0), -1), ((-1.0, 0.0), -1),
               ((1.0, 0.0), 1), ((1.0, 2.0), 1), ((2.0, 1.0), 1)]
    weights = [0.0, 0.0]
    bias = 0.0
    learning_rate = 0.05
    regularization = 0.02
    for _ in range(500):
        gradient = [regularization * value for value in weights]
        bias_gradient = 0.0
        for point, label in samples:
            margin = label * (sum(w * x for w, x in zip(weights, point)) + bias)
            if margin < 1.0:
                for index in range(2):
                    gradient[index] -= label * point[index] / len(samples)
                bias_gradient -= label / len(samples)
        weights = [w - learning_rate * g for w, g in zip(weights, gradient)]
        bias -= learning_rate * bias_gradient
    margins = [label * (sum(w * x for w, x in zip(weights, point)) + bias)
               for point, label in samples]
    print("w={} b={:.3f}".format([round(v, 3) for v in weights], bias))
    print("训练样本间隔：{}".format([round(value, 2) for value in margins]))


if __name__ == "__main__":
    main()
