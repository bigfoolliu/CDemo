"""在共享 CSV 划分和训练集标准化上比较四种分类器。"""

import csv
from math import dist, exp
from pathlib import Path


def balanced_accuracy(labels: list[int], predictions: list[int]) -> float:
    """计算二分类的平衡准确率。

    :param labels: 真实 0/1 标签。
    :param predictions: 与标签逐项对应的预测类别。
    :return: 正类召回率与负类召回率的平均值。
    """
    recalls = []
    for category in (0, 1):
        indices = [i for i, label in enumerate(labels) if label == category]
        if not indices:
            return 0.0
        recalls.append(sum(predictions[i] == category for i in indices) / len(indices))
    return sum(recalls) / 2.0


def fit_logistic(samples: list[tuple[float, float]], labels: list[int],
                 penalty: float) -> tuple[list[float], float]:
    """用全批量梯度下降拟合带 L2 权重惩罚的逻辑回归。

    :param samples: 由训练集统计量标准化的二维训练特征。
    :param labels: 训练集二元标签。
    :param penalty: 非负 L2 正则强度。
    :return: 两个权重和一个偏置。
    """
    weights = [0.0, 0.0]
    bias = 0.0
    for _ in range(400):
        gradient = [penalty * value for value in weights]
        bias_gradient = 0.0
        for features, label in zip(samples, labels):
            logit = sum(w * x for w, x in zip(weights, features)) + bias
            probability = 1.0 / (1.0 + exp(-max(-30.0, min(30.0, logit))))
            for feature in range(2):
                gradient[feature] += (probability - label) * features[feature] / len(samples)
            bias_gradient += (probability - label) / len(samples)
        weights = [w - 0.1 * g for w, g in zip(weights, gradient)]
        bias -= 0.1 * bias_gradient
    return weights, bias


def main() -> None:
    """用验证集选逻辑回归和 K 值，再统一评估独立测试集。

    :return: 无。
    """
    path = Path("data/ml/study_outcomes.csv")
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    feature_names = ("study_hours", "practice_questions")
    train_rows = [row for row in rows if row["split"] == "train"]
    validation_rows = [row for row in rows if row["split"] == "validation"]
    test_rows = [row for row in rows if row["split"] == "test"]
    means = [sum(float(row[name]) for row in train_rows) / len(train_rows)
             for name in feature_names]
    scales = [(sum((float(row[name]) - means[i]) ** 2 for row in train_rows) /
               len(train_rows)) ** 0.5 for i, name in enumerate(feature_names)]
    scales = [scale if scale > 1e-12 else 1.0 for scale in scales]

    def transform(row: dict[str, str]) -> tuple[float, float]:
        """只使用训练集均值和标准差变换一行数据。

        :param row: CSV 中的一个样本记录。
        :return: 标准化的两个特征。
        """
        return tuple((float(row[name]) - means[i]) / scales[i]
                     for i, name in enumerate(feature_names))

    train_x = [transform(row) for row in train_rows]
    train_y = [int(row["passed"]) for row in train_rows]
    validation_x = [transform(row) for row in validation_rows]
    validation_y = [int(row["passed"]) for row in validation_rows]
    test_x = [transform(row) for row in test_rows]
    test_y = [int(row["passed"]) for row in test_rows]

    candidates = []
    for penalty in (0.0, 0.1, 1.0):
        weights, bias = fit_logistic(train_x, train_y, penalty)
        predictions = [int(sum(w * x for w, x in zip(weights, features)) + bias >= 0.0)
                       for features in validation_x]
        candidates.append((balanced_accuracy(validation_y, predictions), penalty, weights, bias))
    _, selected_penalty, logistic_weights, logistic_bias = max(candidates, key=lambda item: item[0])

    knn_candidates = []
    for neighbor_count in (1, 3, 5):
        predictions = []
        for point in validation_x:
            nearest = sorted((dist(point, sample), label)
                             for sample, label in zip(train_x, train_y))[:neighbor_count]
            predictions.append(int(sum(label for _, label in nearest) * 2 > neighbor_count))
        knn_candidates.append((balanced_accuracy(validation_y, predictions), neighbor_count))
    _, selected_k = max(knn_candidates, key=lambda item: item[0])

    stump_candidates = []
    for feature in range(2):
        values = sorted({sample[feature] for sample in train_x})
        for left, right in zip(values, values[1:]):
            threshold = (left + right) / 2.0
            leaves = [[label for sample, label in zip(train_x, train_y)
                       if (sample[feature] <= threshold) == (side == 0)] for side in (0, 1)]
            impurity = sum(len(leaf) / len(train_y) *
                           (1.0 - sum((leaf.count(label) / len(leaf)) ** 2
                                      for label in (0, 1)))
                           for leaf in leaves if leaf)
            stump_candidates.append((impurity, feature, threshold))
    _, stump_feature, stump_threshold = min(stump_candidates)
    stump_leaves = [[label for sample, label in zip(train_x, train_y)
                     if (sample[stump_feature] <= stump_threshold) == (side == 0)]
                    for side in (0, 1)]
    stump_labels = [int(sum(leaf) * 2 > len(leaf)) for leaf in stump_leaves]

    majority_label = int(sum(train_y) * 2 > len(train_y))
    test_predictions = {
        "多数类基线": [majority_label] * len(test_x),
        "逻辑回归": [int(sum(w * x for w, x in zip(logistic_weights, point)) + logistic_bias >= 0.0)
                     for point in test_x],
        "KNN(k={})".format(selected_k): [
            int(sum(label for _, label in sorted((dist(point, sample), label)
                for sample, label in zip(train_x, train_y))[:selected_k]) * 2 > selected_k)
            for point in test_x],
        "决策树桩": [stump_labels[0 if point[stump_feature] <= stump_threshold else 1]
                    for point in test_x],
    }
    print("CSV={} 划分 train/validation/test={}/{}/{}".format(
        path, len(train_x), len(validation_x), len(test_x)))
    print("预处理均值={} 标准差={}；选择 lambda={}、k={}".format(
        [round(value, 3) for value in means], [round(value, 3) for value in scales],
        selected_penalty, selected_k))
    for name, predictions in test_predictions.items():
        accuracy = sum(a == b for a, b in zip(predictions, test_y)) / len(test_y)
        balanced = balanced_accuracy(test_y, predictions)
        print("{}: accuracy={:.1%}, balanced_accuracy={:.1%}".format(name, accuracy, balanced))
    print("所有模型共享数据划分和训练集标准化参数；测试集不参与选参。")


if __name__ == "__main__":
    main()
