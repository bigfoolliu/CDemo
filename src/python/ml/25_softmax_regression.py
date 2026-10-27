"""手写 Softmax 回归，在二维三分类数据上进行全批量梯度下降。"""

from math import exp


def probabilities(scores: list[float]) -> list[float]:
    """对类别分数进行数值稳定的 Softmax 归一化。

    :param scores: 各类别的有限实数分数。
    :return: 总和为 1 的类别概率。
    """
    maximum = max(scores)
    values = [exp(max(-60.0, score - maximum)) for score in scores]
    return [value / sum(values) for value in values]


def main() -> None:
    """训练线性 Softmax 分类器并输出损失和预测类别。

    :return: 无。
    """
    samples = [((-2.0, 0.0), 0), ((-1.5, 0.5), 0), ((-2.0, 1.0), 0),
               ((2.0, 0.0), 1), ((1.5, 0.5), 1), ((2.0, 1.0), 1),
               ((0.0, 2.0), 2), ((0.5, 1.5), 2), ((-0.5, 1.5), 2)]
    weights = [[0.0, 0.0] for _ in range(3)]
    biases = [0.0] * 3
    for _ in range(600):
        gw = [[0.0, 0.0] for _ in range(3)]
        gb = [0.0] * 3
        for features, label in samples:
            probs = probabilities([sum(w * x for w, x in zip(row, features)) + b
                                   for row, b in zip(weights, biases)])
            for category in range(3):
                error = probs[category] - int(category == label)
                for feature in range(2):
                    gw[category][feature] += error * features[feature] / len(samples)
                gb[category] += error / len(samples)
        for category in range(3):
            weights[category] = [w - 0.1 * g for w, g in zip(weights[category], gw[category])]
            biases[category] -= 0.1 * gb[category]
    correct = 0
    for features, label in samples:
        probs = probabilities([sum(w * x for w, x in zip(row, features)) + b
                               for row, b in zip(weights, biases)])
        correct += max(range(3), key=probs.__getitem__) == label
    print("三分类训练准确率={}/{}，权重矩阵={}".format(
        correct, len(samples), [[round(v, 2) for v in row] for row in weights]))


if __name__ == "__main__":
    main()
