"""用距离排序和多数投票实现 K 近邻分类。"""

from math import hypot
from typing import List, Tuple

Sample = Tuple[float, float, int]


def classify(samples: List[Sample], point: Tuple[float, float], k: int) -> int:
    """按最近的 k 个带标签样本预测类别。

    :param samples: (x, y, label) 组成的训练样本列表。
    :param point: 待分类的二维坐标。
    :param k: 邻居数量，须介于 1 和样本数之间。
    :return: 多数票类别，票数相同时返回较小类别。
    :raises ValueError: k 超出有效范围时抛出。
    """
    if not 1 <= k <= len(samples):
        raise ValueError("k 必须在 1 到样本数之间")
    ranked = sorted(
        ((hypot(x - point[0], y - point[1]), index, label)
         for index, (x, y, label) in enumerate(samples)),
        key=lambda item: (item[0], item[1]),
    )
    positive = sum(item[2] == 1 for item in ranked[:k])
    return int(positive > k - positive)


def main() -> None:
    """用固定六点验证 K 近邻分类和票数平局。

    :return: 无。
    """
    samples = [(0, 0, 0), (0, 1, 0), (1, 0, 0), (4, 4, 1), (4, 5, 1), (5, 4, 1)]
    for point in ((0.2, 0.3), (4.2, 4.3), (2.0, 2.0)):
        print("KNN：{}，k=3 → 类别 {}".format(point, classify(samples, point, 3)))


if __name__ == "__main__":
    main()
