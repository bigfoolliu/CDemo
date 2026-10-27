"""用一维回归树桩逐轮拟合平方损失残差，演示梯度提升。"""


def fit_stump(xs: list[float], residuals: list[float]) -> tuple[float, float, float]:
    """枚举切分点并返回平方误差最小的回归树桩。

    :param xs: 一维输入特征。
    :param residuals: 当前模型的残差目标。
    :return: 切分阈值、左叶均值、右叶均值。
    """
    best = (float("inf"), 0.0, 0.0, 0.0)
    ordered = sorted(set(xs))
    for left, right in zip(ordered, ordered[1:]):
        threshold = (left + right) / 2.0
        low = [r for x, r in zip(xs, residuals) if x <= threshold]
        high = [r for x, r in zip(xs, residuals) if x > threshold]
        low_mean = sum(low) / len(low)
        high_mean = sum(high) / len(high)
        error = sum((value - low_mean) ** 2 for value in low)
        error += sum((value - high_mean) ** 2 for value in high)
        best = min(best, (error, threshold, low_mean, high_mean))
    return best[1], best[2], best[3]


def main() -> None:
    """累加多棵树桩并打印每轮训练平方误差。

    :return: 无。
    """
    xs = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    targets = [0.0, 0.2, 0.1, 1.0, 1.2, 1.1]
    predictions = [sum(targets) / len(targets)] * len(xs)
    rate = 0.5
    for iteration in range(5):
        residuals = [target - prediction for target, prediction in zip(targets, predictions)]
        threshold, left, right = fit_stump(xs, residuals)
        predictions = [prediction + rate * (left if x <= threshold else right)
                       for x, prediction in zip(xs, predictions)]
        mse = sum((target - prediction) ** 2 for target, prediction in zip(targets, predictions)) / len(xs)
        print("轮次 {}: threshold={:.1f}, 训练 MSE={:.5f}".format(iteration + 1, threshold, mse))


if __name__ == "__main__":
    main()
