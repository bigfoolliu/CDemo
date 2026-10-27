"""用纯 Python 实现 2→4→1 XOR 多层感知机和反向传播。"""

from math import exp, tanh
from random import Random
from typing import List, Tuple

DATA: List[Tuple[Tuple[float, float], float]] = [
    ((0.0, 0.0), 0.0), ((0.0, 1.0), 1.0), ((1.0, 0.0), 1.0), ((1.0, 1.0), 0.0)
]


def main() -> None:
    """训练 XOR 网络并显示四种输入的预测。

    :return: 无。
    """
    rng = Random(7)
    hidden = 4
    w1 = [[rng.uniform(-0.8, 0.8) for _ in range(2)] for _ in range(hidden)]
    b1 = [0.0] * hidden
    w2 = [rng.uniform(-0.8, 0.8) for _ in range(hidden)]
    b2 = 0.0

    for _ in range(5000):
        gw1 = [[0.0, 0.0] for _ in range(hidden)]
        gb1 = [0.0] * hidden
        gw2 = [0.0] * hidden
        gb2 = 0.0
        for inputs, target in DATA:
            activation = [tanh(sum(w * x for w, x in zip(row, inputs)) + bias)
                          for row, bias in zip(w1, b1)]
            logit = sum(w * h for w, h in zip(w2, activation)) + b2
            probability = 1.0 / (1.0 + exp(-max(-30.0, min(30.0, logit))))
            output_delta = probability - target
            gb2 += output_delta
            for unit in range(hidden):
                gw2[unit] += output_delta * activation[unit]
                hidden_delta = output_delta * w2[unit] * (1.0 - activation[unit] ** 2)
                gb1[unit] += hidden_delta
                for feature in range(2):
                    gw1[unit][feature] += hidden_delta * inputs[feature]

        rate = 0.1 / len(DATA)
        for unit in range(hidden):
            b1[unit] -= rate * gb1[unit]
            w2[unit] -= rate * gw2[unit]
            for feature in range(2):
                w1[unit][feature] -= rate * gw1[unit][feature]
        b2 -= rate * gb2

    print("XOR 多层感知机：")
    for inputs, target in DATA:
        activation = [tanh(sum(w * x for w, x in zip(row, inputs)) + bias)
                      for row, bias in zip(w1, b1)]
        logit = sum(w * h for w, h in zip(w2, activation)) + b2
        probability = 1.0 / (1.0 + exp(-max(-30.0, min(30.0, logit))))
        print("  {} → {:.4f}（标签 {:.0f}）".format(inputs, probability, target))


if __name__ == "__main__":
    main()
