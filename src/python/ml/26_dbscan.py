"""使用邻域查询和密度扩展实现二维 DBSCAN 聚类。"""

from math import dist


def main() -> None:
    """按 epsilon 邻域和最小点数识别簇与噪声。

    :return: 无。
    """
    points = [(0.0, 0.0), (0.2, 0.1), (-0.1, 0.2), (0.1, -0.2),
              (3.0, 3.0), (3.2, 3.1), (2.9, 3.2), (8.0, 0.0)]
    epsilon = 0.45
    minimum = 3
    neighbors = [[j for j, other in enumerate(points)
                  if dist(point, other) <= epsilon] for point in points]
    labels = [None] * len(points)
    cluster = 0
    for index in range(len(points)):
        if labels[index] is not None or len(neighbors[index]) < minimum:
            continue
        labels[index] = cluster
        queue = list(neighbors[index])
        cursor = 0
        while cursor < len(queue):
            candidate = queue[cursor]
            cursor += 1
            if labels[candidate] == -1:
                labels[candidate] = cluster
            if labels[candidate] is not None:
                continue
            labels[candidate] = cluster
            if len(neighbors[candidate]) >= minimum:
                queue.extend(item for item in neighbors[candidate] if item not in queue)
        cluster += 1
    labels = [-1 if label is None else label for label in labels]
    print("epsilon={} min_points={} 聚类标签={}".format(epsilon, minimum, labels))
    print("-1 表示噪声；边界点可归入簇，但不会继续扩展邻居。")


if __name__ == "__main__":
    main()
