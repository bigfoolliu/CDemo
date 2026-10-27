"""从头训练一个小型去噪自编码器，并观察瓶颈表示与重建误差。"""

from math import exp
from random import Random


def sigmoid(value: float) -> float:
    """将 logit 映射到概率，并限制指数输入范围。

    :param value: 实数 logit。
    :return: 0 到 1 之间的概率。
    """
    value = max(-30.0, min(30.0, value))
    return 1.0 / (1.0 + exp(-value))


def main() -> None:
    """训练 8→3→8 网络，让损坏输入重建为干净二值图案。

    :return: 无。
    """
    patterns = ([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    rng = Random(9)
    hidden_count = 3
    encoder = [[rng.uniform(-0.5, 0.5) for _ in range(8)] for _ in range(hidden_count)]
    encoder_bias = [0.0] * hidden_count
    decoder = [[rng.uniform(-0.5, 0.5) for _ in range(hidden_count)] for _ in range(8)]
    decoder_bias = [0.0] * 8
    rate = 0.4

    for epoch in range(500):
        for index, clean in enumerate(patterns):
            noisy = list(clean)
            noise_index = (index + epoch) % len(noisy)
            noisy[noise_index] = 1.0 - noisy[noise_index]
            hidden = [sigmoid(sum(w * value for w, value in zip(row, noisy)) + bias)
                      for row, bias in zip(encoder, encoder_bias)]
            output = [sigmoid(sum(w * value for w, value in zip(row, hidden)) + bias)
                      for row, bias in zip(decoder, decoder_bias)]
            output_delta = [prediction - target for prediction, target in zip(output, clean)]
            hidden_delta = [hidden[j] * (1.0 - hidden[j]) * sum(
                decoder[i][j] * output_delta[i] for i in range(8)) for j in range(hidden_count)]
            for i in range(8):
                for j in range(hidden_count):
                    decoder[i][j] -= rate * output_delta[i] * hidden[j]
                decoder_bias[i] -= rate * output_delta[i]
            for j in range(hidden_count):
                for i in range(8):
                    encoder[j][i] -= rate * hidden_delta[j] * noisy[i]
                encoder_bias[j] -= rate * hidden_delta[j]

    total_error = 0.0
    for index, clean in enumerate(patterns):
        noisy = list(clean)
        noise_index = (index + 1) % len(noisy)
        noisy[noise_index] = 1.0 - noisy[noise_index]
        hidden = [sigmoid(sum(w * value for w, value in zip(row, noisy)) + bias)
                  for row, bias in zip(encoder, encoder_bias)]
        output = [sigmoid(sum(w * value for w, value in zip(row, hidden)) + bias)
                  for row, bias in zip(decoder, decoder_bias)]
        total_error += sum((prediction - target) ** 2 for prediction, target in zip(output, clean)) / 8
        print("输入={} 目标={} 重建={}".format(
            [int(value) for value in noisy], [int(value) for value in clean],
            [int(value >= 0.5) for value in output]))
    print("独立固定损坏样本平均重建 MSE={:.4f}".format(total_error / len(patterns)))


if __name__ == "__main__":
    main()
