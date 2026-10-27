"""用小型 ReLU 多层感知机学习二维内圆盘与外圆环。"""

from math import cos, exp, pi, sin, sqrt
from random import Random
from typing import List, Tuple

Point = Tuple[float, float, float]


def make_data(seed: int, count: int) -> List[Point]:
    """生成平衡的内圆盘和外圆环样本。

    :param seed: 可重复的数据随机种子。
    :param count: 总样本数，须为偶数。
    :return: (x, y, label) 样本列表。
    """
    rng = Random(seed)
    samples = []
    for index in range(count):
        label = float(index % 2)
        low, high = (0.0, 0.55) if label == 1.0 else (0.75, 1.0)
        radius = sqrt(rng.uniform(low * low, high * high))
        angle = 2.0 * pi * rng.random()
        samples.append((radius * cos(angle), radius * sin(angle), label))
    return samples


def main() -> None:
    """训练两层 ReLU 网络，并比较训练集和固定探测点。

    :return: 无。
    """
    rng = Random(31)
    width = 8
    w1 = [[rng.uniform(-0.8, 0.8) for _ in range(2)] for _ in range(width)]
    b1 = [0.0] * width
    w2 = [[rng.uniform(-0.5, 0.5) for _ in range(width)] for _ in range(width)]
    b2 = [0.0] * width
    w3 = [rng.uniform(-0.5, 0.5) for _ in range(width)]
    b3 = 0.0
    samples = make_data(11, 64)

    for epoch in range(250):
        rng.shuffle(samples)
        for x, y, target in samples:
            z1 = [sum(w * value for w, value in zip(row, (x, y))) + bias
                  for row, bias in zip(w1, b1)]
            h1 = [max(0.0, value) for value in z1]
            z2 = [sum(w * value for w, value in zip(row, h1)) + bias
                  for row, bias in zip(w2, b2)]
            h2 = [max(0.0, value) for value in z2]
            logit = sum(w * value for w, value in zip(w3, h2)) + b3
            probability = 1.0 / (1.0 + exp(-max(-30.0, min(30.0, logit))))
            delta3 = probability - target
            grad3 = [delta3 * value for value in h2]
            delta2 = [delta3 * w3[i] * (z2[i] > 0.0) for i in range(width)]
            delta1 = [sum(delta2[j] * w2[j][i] for j in range(width)) * (z1[i] > 0.0)
                      for i in range(width)]
            rate = 0.02
            for i in range(width):
                w3[i] -= rate * grad3[i]
                b2[i] -= rate * delta2[i]
                for j in range(width):
                    w2[i][j] -= rate * delta2[i] * h1[j]
                b1[i] -= rate * delta1[i]
                w1[i][0] -= rate * delta1[i] * x
                w1[i][1] -= rate * delta1[i] * y
            b3 -= rate * delta3

    probes = [(0.1, 0.1), (0.9, 0.0), (0.0, 0.95), (0.5, 0.5)]
    print("二维 MLP：内圆盘概率高，外圆环概率低")
    for x, y in probes:
        h1 = [max(0.0, sum(w * value for w, value in zip(row, (x, y))) + bias)
              for row, bias in zip(w1, b1)]
        h2 = [max(0.0, sum(w * value for w, value in zip(row, h1)) + bias)
              for row, bias in zip(w2, b2)]
        score = sum(w * value for w, value in zip(w3, h2)) + b3
        print("  ({:.2f}, {:.2f}) → {:.3f}".format(
            x, y, 1.0 / (1.0 + exp(-max(-30.0, min(30.0, score))))))


if __name__ == "__main__":
    main()
