"""用轮转分折比较两个固定超参数，说明验证数据只用于选参。"""

from math import exp, log


def main() -> None:
    """执行三折评估并报告候选模型的平均验证损失。

    :return: 无。
    """
    samples = [(-2.0, 0), (-1.5, 0), (-1.0, 0), (-0.3, 0),
               (0.3, 1), (1.0, 1), (1.5, 1), (2.0, 1), (2.5, 1)]
    candidates = (0.0, 0.2)
    for penalty in candidates:
        losses = []
        for fold in range(3):
            train = [item for index, item in enumerate(samples) if index % 3 != fold]
            valid = [item for index, item in enumerate(samples) if index % 3 == fold]
            weight = 0.0
            for _ in range(300):
                gradient = penalty * weight
                for x, label in train:
                    probability = 1.0 / (1.0 + exp(-max(-30.0, min(30.0, weight * x))))
                    gradient += (probability - label) * x / len(train)
                weight -= 0.1 * gradient
            loss = 0.0
            for x, label in valid:
                probability = 1.0 / (1.0 + exp(-max(-30.0, min(30.0, weight * x))))
                loss -= label * log(max(probability, 1e-12))
                loss -= (1 - label) * log(max(1 - probability, 1e-12))
            losses.append(loss / len(valid))
        print("L2={} 各折损失={} 平均={:.4f}".format(
            penalty, [round(value, 4) for value in losses], sum(losses) / len(losses)))
    print("交叉验证用于选超参数；最终测试集应保持未参与选择。")


if __name__ == "__main__":
    main()
