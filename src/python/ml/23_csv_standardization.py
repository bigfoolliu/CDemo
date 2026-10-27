"""读取仓库 CSV，使用训练集统计量标准化并独立评估测试集。"""

import csv
from math import exp
from pathlib import Path


def main() -> None:
    """拟合逻辑回归并分别报告训练、验证和测试准确率。

    :return: 无。
    """
    path = Path("data/ml/study_outcomes.csv")
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    train = [row for row in rows if row["split"] == "train"]
    mean = [sum(float(row[key]) for row in train) / len(train)
            for key in ("study_hours", "practice_questions")]
    scale = [max(1e-12, (sum((float(row[key]) - mean[i]) ** 2 for row in train) /
                         len(train)) ** 0.5)
             for i, key in enumerate(("study_hours", "practice_questions"))]
    samples = [(tuple((float(row[key]) - mean[i]) / scale[i]
                      for i, key in enumerate(("study_hours", "practice_questions"))),
                int(row["passed"]), row["split"]) for row in rows]
    weights = [0.0, 0.0]
    bias = 0.0
    for _ in range(1200):
        gw = [0.0, 0.0]
        gb = 0.0
        for features, label, split in samples:
            if split != "train":
                continue
            probability = 1.0 / (1.0 + exp(-max(-30.0, min(30.0,
                sum(w * x for w, x in zip(weights, features)) + bias))))
            for i in range(2):
                gw[i] += (probability - label) * features[i] / len(train)
            gb += (probability - label) / len(train)
        weights = [w - 0.15 * g for w, g in zip(weights, gw)]
        bias -= 0.15 * gb
    print("CSV={} 样本数={} 训练均值={} 标准差={}".format(path, len(rows), mean, scale))
    for split in ("train", "validation", "test"):
        subset = [sample for sample in samples if sample[2] == split]
        correct = 0
        for features, label, _ in subset:
            score = sum(w * x for w, x in zip(weights, features)) + bias
            correct += (score >= 0.0) == bool(label)
        print("{}: {}/{} 正确，准确率={:.1%}".format(
            split, correct, len(subset), correct / len(subset)))
    print("均值和标准差只从 train 计算，防止验证/测试信息泄漏。")


if __name__ == "__main__":
    main()
