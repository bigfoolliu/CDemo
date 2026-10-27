"""用基尼增益递归构造一个带深度限制的二维分类决策树。"""

from itertools import combinations
from typing import Any

Sample = tuple[tuple[float, float], int]
Tree = Any


def gini(labels: list[int]) -> float:
    """计算标签列表的基尼不纯度。

    :param labels: 非空的 0/1 标签列表。
    :return: 纯节点为 0 的基尼不纯度。
    """
    if not labels:
        return 0.0
    return 1.0 - sum((labels.count(label) / len(labels)) ** 2 for label in (0, 1))


def best_split(samples: list[Sample]) -> tuple[float, int, float] | None:
    """遍历特征阈值并返回加权基尼最小的切分。

    :param samples: 当前节点中的特征与标签。
    :return: 加权不纯度、特征索引、阈值；没有有效切分时返回 None。
    """
    parent_impurity = gini([label for _, label in samples])
    best: tuple[float, int, float] | None = None
    for feature in range(2):
        values = sorted({point[feature] for point, _ in samples})
        for lower, upper in combinations(values, 2):
            threshold = (lower + upper) / 2.0
            left = [label for point, label in samples if point[feature] <= threshold]
            right = [label for point, label in samples if point[feature] > threshold]
            if not left or not right:
                continue
            impurity = (len(left) / len(samples) * gini(left) +
                        len(right) / len(samples) * gini(right))
            candidate = (impurity, feature, threshold)
            if best is None or candidate < best:
                best = candidate
    if best is None or best[0] >= parent_impurity:
        return None
    return best


def build_tree(samples: list[Sample], depth: int) -> Tree:
    """递归生长基尼增益为正的二叉分类树。

    :param samples: 当前节点的非空训练样本。
    :param depth: 剩余最大切分深度，非正数时生成叶节点。
    :return: 整数类别叶节点，或含特征、阈值和子树的元组。
    """
    labels = [label for _, label in samples]
    majority = int(sum(labels) * 2 > len(labels))
    if depth <= 0 or all(label == labels[0] for label in labels):
        return majority
    split = best_split(samples)
    if split is None:
        return majority
    _, feature, threshold = split
    left = [sample for sample in samples if sample[0][feature] <= threshold]
    right = [sample for sample in samples if sample[0][feature] > threshold]
    if not left or not right:
        return majority
    return (feature, threshold, build_tree(left, depth - 1),
            build_tree(right, depth - 1))


def predict(tree: Tree, point: tuple[float, float]) -> int:
    """沿树的切分条件遍历到叶节点并返回类别。

    :param tree: build_tree 返回的只读树。
    :param point: 待分类二维坐标。
    :return: 预测类别 0 或 1。
    """
    while not isinstance(tree, int):
        feature, threshold, left, right = tree
        tree = left if point[feature] <= threshold else right
    return tree


def main() -> None:
    """训练深度受限的树，并输出训练点和探测点的预测。

    :return: 无。
    """
    samples: list[Sample] = [
        ((1.0, 1.0), 0), ((1.5, 2.0), 0), ((2.0, 1.0), 0),
        ((4.0, 4.0), 1), ((4.5, 5.0), 1), ((5.0, 4.0), 1),
    ]
    tree = build_tree(samples, depth=3)
    correct = sum(predict(tree, point) == label for point, label in samples)
    probes = ((1.8, 1.7), (4.2, 4.5))
    print("递归树结构：{}".format(tree))
    print("训练准确率={}/{}".format(correct, len(samples)))
    print("探测点预测={}".format([predict(tree, point) for point in probes]))


if __name__ == "__main__":
    main()
