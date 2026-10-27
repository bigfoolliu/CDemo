"""计算 GRU 单元的门值，观察更新门如何保留旧记忆。"""

from math import exp, tanh


def sigmoid(value: float) -> float:
    """计算 sigmoid 激活。

    :param value: 实数输入。
    :return: 0 到 1 之间的结果。
    """
    return 1.0 / (1.0 + exp(-max(-40.0, min(40.0, value))))


def main() -> None:
    """对一段含干扰位的输入序列执行标量 GRU 前向传播。

    :return: 无。
    """
    sequence = [1.0, 0.0, -0.3, 0.2, 0.0, -0.1]
    hidden = 0.0
    for step, value in enumerate(sequence):
        update = sigmoid(-1.2 * value + 1.5)
        reset = sigmoid(0.8 * value - 0.2)
        candidate = tanh(0.9 * value + 0.7 * reset * hidden)
        hidden = update * hidden + (1.0 - update) * candidate
        print("t={} x={:+.1f} z={:.3f} r={:.3f} h={:+.3f}".format(
            step, value, update, reset, hidden))
    print("更新门 z 接近 1 时保留旧状态；接近 0 时更多写入候选状态。")


if __name__ == "__main__":
    main()
