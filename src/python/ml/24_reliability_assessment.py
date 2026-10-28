"""计算不平衡分类的混淆矩阵、Brier 分数与分箱校准误差。"""


def main() -> None:
    """对固定概率预测评估阈值指标及概率校准情况。

    :return: 无。
    """
    labels = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
    probabilities = [0.05, 0.12, 0.18, 0.22, 0.31, 0.42, 0.48, 0.55, 0.72, 0.91]
    threshold = 0.4
    predictions = [int(value >= threshold) for value in probabilities]
    tp = sum(y == 1 and p == 1 for y, p in zip(labels, predictions))
    fp = sum(y == 0 and p == 1 for y, p in zip(labels, predictions))
    fn = sum(y == 1 and p == 0 for y, p in zip(labels, predictions))
    tn = len(labels) - tp - fp - fn
    brier = sum((p - y) ** 2 for y, p in zip(labels, probabilities)) / len(labels)
    ece = 0.0
    for lower in (0.0, 0.5):
        bucket = [(y, p) for y, p in zip(labels, probabilities)
                  if lower <= p < lower + 0.5 or (lower == 0.5 and p == 1.0)]
        if bucket:
            confidence = sum(p for _, p in bucket) / len(bucket)
            frequency = sum(y for y, _ in bucket) / len(bucket)
            ece += len(bucket) / len(labels) * abs(confidence - frequency)
    print("阈值={:.2f} 混淆矩阵 TN={} FP={} FN={} TP={}".format(threshold, tn, fp, fn, tp))
    print("Brier={:.4f} 两箱 ECE={:.4f} 正类基准率={:.1%}".format(
        brier, ece, sum(labels) / len(labels)))
    print("阈值改变分类决策；Brier/ECE 检查概率质量，准确率不能替代它们。")
    print("以上是固定样本上的点估计，没有计算置信区间；可靠性示例见 M10 教程。")


if __name__ == "__main__":
    main()
