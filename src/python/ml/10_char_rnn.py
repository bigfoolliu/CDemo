"""手写字符级 RNN 的前向传播、softmax 损失和完整反向传播。"""

from math import exp, tanh
from random import Random


def softmax(values: list[float]) -> list[float]:
    """将类别 logits 稳定地转换为概率。

    :param values: 一组有限实数分数。
    :return: 总和为 1 的概率列表。
    """
    maximum = max(values)
    values = [exp(max(-40.0, value - maximum)) for value in values]
    total = sum(values)
    return [value / total for value in values]


def main() -> None:
    """训练小型字符 RNN 预测下一个字符，并贪心生成文本。

    :return: 无。
    """
    text = "abc abc cab abc cab abc "
    alphabet = "abc "
    sequence = [alphabet.index(character) for character in text]
    vocabulary = len(alphabet)
    hidden_size = 6
    rng = Random(10)
    input_weights = [[rng.uniform(-0.3, 0.3) for _ in range(vocabulary)]
                     for _ in range(hidden_size)]
    recurrent_weights = [[rng.uniform(-0.3, 0.3) for _ in range(hidden_size)]
                         for _ in range(hidden_size)]
    hidden_bias = [0.0] * hidden_size
    output_weights = [[rng.uniform(-0.3, 0.3) for _ in range(hidden_size)]
                      for _ in range(vocabulary)]
    output_bias = [0.0] * vocabulary

    for _ in range(120):
        states = [[0.0] * hidden_size]
        probabilities = []
        for position in range(len(sequence) - 1):
            current = sequence[position]
            state = [tanh(input_weights[h][current] + hidden_bias[h] +
                          sum(recurrent_weights[h][j] * states[-1][j]
                              for j in range(hidden_size))) for h in range(hidden_size)]
            states.append(state)
            logits = [sum(output_weights[c][h] * state[h] for h in range(hidden_size)) + output_bias[c]
                      for c in range(vocabulary)]
            probabilities.append(softmax(logits))

        grad_input = [[0.0] * vocabulary for _ in range(hidden_size)]
        grad_recurrent = [[0.0] * hidden_size for _ in range(hidden_size)]
        grad_hidden_bias = [0.0] * hidden_size
        grad_output = [[0.0] * hidden_size for _ in range(vocabulary)]
        grad_output_bias = [0.0] * vocabulary
        next_state_gradient = [0.0] * hidden_size
        for position in range(len(probabilities) - 1, -1, -1):
            target = sequence[position + 1]
            output_delta = [probabilities[position][c] - int(c == target)
                            for c in range(vocabulary)]
            for c in range(vocabulary):
                grad_output_bias[c] += output_delta[c]
                for h in range(hidden_size):
                    grad_output[c][h] += output_delta[c] * states[position + 1][h]
            state_gradient = [sum(output_delta[c] * output_weights[c][h]
                                  for c in range(vocabulary)) + next_state_gradient[h]
                              for h in range(hidden_size)]
            state_delta = [state_gradient[h] * (1.0 - states[position + 1][h] ** 2)
                           for h in range(hidden_size)]
            for h in range(hidden_size):
                grad_hidden_bias[h] += state_delta[h]
                grad_input[h][sequence[position]] += state_delta[h]
                for j in range(hidden_size):
                    grad_recurrent[h][j] += state_delta[h] * states[position][j]
            next_state_gradient = [sum(state_delta[h] * recurrent_weights[h][j]
                                       for h in range(hidden_size)) for j in range(hidden_size)]

        rate = 0.08 / len(sequence)
        for h in range(hidden_size):
            hidden_bias[h] -= rate * grad_hidden_bias[h]
            for c in range(vocabulary):
                input_weights[h][c] -= rate * grad_input[h][c]
            for j in range(hidden_size):
                recurrent_weights[h][j] -= rate * grad_recurrent[h][j]
        for c in range(vocabulary):
            output_bias[c] -= rate * grad_output_bias[c]
            for h in range(hidden_size):
                output_weights[c][h] -= rate * grad_output[c][h]

    generated = [alphabet.index("a")]
    state = [0.0] * hidden_size
    for _ in range(24):
        current = generated[-1]
        state = [tanh(input_weights[h][current] + hidden_bias[h] +
                      sum(recurrent_weights[h][j] * state[j] for j in range(hidden_size)))
                 for h in range(hidden_size)]
        logits = [sum(output_weights[c][h] * state[h] for h in range(hidden_size)) + output_bias[c]
                  for c in range(vocabulary)]
        generated.append(max(range(vocabulary), key=softmax(logits).__getitem__))
    print("训练序列：{}".format(repr(text)))
    print("RNN 贪心生成：{}".format("".join(alphabet[index] for index in generated)))


if __name__ == "__main__":
    main()
