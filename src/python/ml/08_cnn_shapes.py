"""用固定卷积核提取合成图形特征，再训练线性分类头。"""

from math import exp


def make_image(kind: int) -> list[list[float]]:
    """生成方形、竖线或横线的 8×8 灰度图。

    :param kind: 图形编号，取 0、1 或 2。
    :return: 像素值为 0 或 1 的二维列表。
    """
    image = [[0.0] * 8 for _ in range(8)]
    if kind == 0:
        for row in range(2, 6):
            for column in range(2, 6):
                image[row][column] = 1.0
    elif kind == 1:
        for row in range(1, 7):
            image[row][3] = 1.0
            image[row][4] = 1.0
    else:
        for column in range(1, 7):
            image[3][column] = 1.0
            image[4][column] = 1.0
    return image


def convolve(image: list[list[float]], kernel: list[list[float]]) -> list[list[float]]:
    """对图像执行步长为 1、无填充的二维卷积相关运算。

    :param image: 输入灰度图。
    :param kernel: 3×3 特征核。
    :return: ReLU 激活后的特征图。
    """
    result = []
    for row in range(len(image) - 2):
        output_row = []
        for column in range(len(image[0]) - 2):
            value = sum(
                image[row + ky][column + kx] * kernel[ky][kx]
                for ky in range(3) for kx in range(3)
            )
            output_row.append(max(0.0, value))
        result.append(output_row)
    return result


def extract_features(kind: int) -> list[float]:
    """提取三个卷积核经 ReLU 和 2×2 最大池化后的最大响应。

    :param kind: 合成图形类别编号。
    :return: 三维特征向量。
    """
    kernels = [
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        [[1, 1, 1], [1, -8, 1], [1, 1, 1]],
    ]
    maps = [convolve(make_image(kind), kernel) for kernel in kernels]
    return [max(maps[feature][r][c] for r in range(0, 6, 2)
                for c in range(0, 6, 2)) for feature in range(len(kernels))]


def softmax(scores: list[float]) -> list[float]:
    """将类别分数转为稳定的概率分布。

    :param scores: 各类别的 logit。
    :return: 总和为 1 的类别概率。
    """
    values = [exp(score - max(scores)) for score in scores]
    return [value / sum(values) for value in values]


def main() -> None:
    """训练 Softmax 分类头，并展示三种图形的预测。

    :return: 无。
    """
    labels = ("方形", "竖线", "横线")
    features = [extract_features(kind) for kind in range(len(labels))]
    weights = [[0.0] * 3 for _ in labels]
    biases = [0.0] * len(labels)
    for _ in range(300):
        grad_weights = [[0.0] * 3 for _ in labels]
        grad_biases = [0.0] * len(labels)
        for label, vector in enumerate(features):
            probs = softmax([sum(w * x for w, x in zip(row, vector)) + bias
                             for row, bias in zip(weights, biases)])
            for category in range(len(labels)):
                error = probs[category] - int(category == label)
                grad_biases[category] += error
                for feature in range(3):
                    grad_weights[category][feature] += error * vector[feature]
        for category in range(len(labels)):
            biases[category] -= 0.03 * grad_biases[category] / len(labels)
            weights[category] = [w - 0.03 * g / len(labels)
                                 for w, g in zip(weights[category], grad_weights[category])]
    for kind, label in enumerate(labels):
        probs = softmax([sum(w * x for w, x in zip(row, features[kind])) + bias
                         for row, bias in zip(weights, biases)])
        prediction = max(range(len(labels)), key=probs.__getitem__)
        print("{}：特征={} 预测={} 概率={:.3f}".format(
            label, [round(value, 1) for value in features[kind]],
            labels[prediction], probs[prediction]))
    print("卷积核固定以突出特征提取；C 案例训练完整 CNN。")


if __name__ == "__main__":
    main()
