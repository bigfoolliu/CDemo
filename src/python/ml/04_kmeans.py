"""用分配与中心更新实现二维 K-means 聚类。"""

from math import hypot
from typing import List, Tuple

Point = Tuple[float, float]


def train(points: List[Point], centers: List[Point]) -> Tuple[List[Point], int]:
    """交替分配样本并更新聚类中心。

    :param points: 固定二维训练点。
    :param centers: 两个初始中心。
    :return: 收敛后的两个中心和迭代次数。
    """
    current = list(centers)
    for iteration in range(1, 101):
        groups: List[List[Point]] = [[], []]
        for point in points:
            cluster = min(range(2), key=lambda index: (hypot(
                point[0] - current[index][0], point[1] - current[index][1]
            ), index))
            groups[cluster].append(point)

        updated = []
        for index, group in enumerate(groups):
            if group:
                updated.append((sum(p[0] for p in group) / len(group),
                                sum(p[1] for p in group) / len(group)))
            else:
                updated.append(current[index])
        movement = sum(hypot(a[0] - b[0], a[1] - b[1]) for a, b in zip(current, updated))
        current = updated
        if movement < 1e-12:
            return current, iteration
    return current, 100


def main() -> None:
    """聚类与 C 示例相同的六个无标签点。

    :return: 无。
    """
    points = [(0, 0), (0, 1), (1, 0), (8, 8), (8, 9), (9, 8)]
    centers, iterations = train(points, [points[0], points[3]])
    print("K-means：{} 轮，中心 0={}，中心 1={}".format(iterations, centers[0], centers[1]))


if __name__ == "__main__":
    main()
