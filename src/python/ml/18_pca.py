"""中心化二维数据，用协方差矩阵幂迭代求主成分。"""

from math import sqrt


def main() -> None:
    """计算第一主成分方向、投影及一维重建误差。

    :return: 无。
    """
    points = [(2.5, 2.4), (0.5, 0.7), (2.2, 2.9), (1.9, 2.2),
              (3.1, 3.0), (2.3, 2.7), (2.0, 1.6), (1.0, 1.1)]
    mean = tuple(sum(point[i] for point in points) / len(points) for i in range(2))
    centered = [(x - mean[0], y - mean[1]) for x, y in points]
    covariance = [[sum(point[i] * point[j] for point in centered) / (len(points) - 1)
                   for j in range(2)] for i in range(2)]
    vector = (1.0, 1.0)
    for _ in range(30):
        next_vector = tuple(sum(covariance[i][j] * vector[j] for j in range(2))
                            for i in range(2))
        norm = sqrt(sum(value * value for value in next_vector))
        vector = tuple(value / norm for value in next_vector)
    projections = [sum(a * b for a, b in zip(point, vector)) for point in centered]
    squared_error = sum(sum((point[j] - value * vector[j]) ** 2 for j in range(2))
                        for point, value in zip(centered, projections)) / len(points)
    print("均值={} 主成分方向={}".format(mean, tuple(round(v, 4) for v in vector)))
    print("一维投影前 3 项={} 平均重建平方误差={:.5f}".format(
        [round(value, 3) for value in projections[:3]], squared_error))


if __name__ == "__main__":
    main()
