"""M01：用固定二维点和基向量观察矩阵变换。"""

from __future__ import annotations

import argparse
import math
from typing import NamedTuple


class Vec2(NamedTuple):
    """有序的二维坐标或向量。"""

    x: float
    y: float


class Mat2(NamedTuple):
    """按行保存的二阶矩阵；每行含两个系数。"""

    first: tuple[float, float]
    second: tuple[float, float]


POINT = Vec2(1.0, 2.0)
SCENES = (
    ("identity", Mat2((1.0, 0.0), (0.0, 1.0))),
    ("scale", Mat2((2.0, 0.0), (0.0, 0.5))),
    ("rotate_ccw_90", Mat2((0.0, -1.0), (1.0, 0.0))),
    ("shear_x", Mat2((1.0, 1.0), (0.0, 1.0))),
    ("project_x", Mat2((1.0, 0.0), (0.0, 0.0))),
    ("rotate_after_scale", Mat2((0.0, -0.5), (2.0, 0.0))),
    ("scale_after_rotate", Mat2((0.0, -2.0), (0.5, 0.0))),
)


def apply(matrix: Mat2, point: Vec2) -> Vec2:
    """计算矩阵作用于列向量后的坐标。

    :param matrix: 行优先的二阶矩阵，元素必须有限。
    :param point: 有限的二维点或向量。
    :return: 有限的变换结果。
    :raises ValueError: 输入或计算结果含非有限数。
    """
    if not all(math.isfinite(value) for row in matrix for value in row):
        raise ValueError("矩阵元素必须有限")
    if not math.isfinite(point.x) or not math.isfinite(point.y):
        raise ValueError("输入点必须有限")
    result = Vec2(matrix.first[0] * point.x + matrix.first[1] * point.y,
                  matrix.second[0] * point.x + matrix.second[1] * point.y)
    if not math.isfinite(result.x) or not math.isfinite(result.y):
        raise ValueError("变换结果必须有限")
    return result


def multiply(left: Mat2, right: Mat2) -> Mat2:
    """复合变换：对列向量先作用 right，再作用 left。

    :param left: 有限矩阵，最后作用。
    :param right: 有限矩阵，最先作用。
    :return: 有限的复合矩阵。
    :raises ValueError: 输入或乘积含非有限数。
    """
    if not all(math.isfinite(value) for matrix in (left, right)
               for row in matrix for value in row):
        raise ValueError("矩阵元素必须有限")
    rows = []
    for row in left:
        result_row = []
        for column in range(2):
            value = row[0] * right.first[column] + row[1] * right.second[column]
            if not math.isfinite(value):
                raise ValueError("复合矩阵元素必须有限")
            result_row.append(value)
        rows.append(tuple(result_row))
    return Mat2(rows[0], rows[1])


def determinant(matrix: Mat2) -> float:
    """计算用于观察有向面积缩放的行列式。

    :param matrix: 元素有限且计算不溢出的二阶矩阵。
    :return: 二阶行列式。
    """
    return matrix.first[0] * matrix.second[1] - matrix.first[1] * matrix.second[0]


def scene_rows() -> list[tuple[str, str, Vec2, float]]:
    """返回与 C 演示使用同一固定输入的输出表。

    :return: 场景、点标签、变换坐标及行列式组成的列表。
    """
    points = (("e1", Vec2(1.0, 0.0)), ("e2", Vec2(0.0, 1.0)), ("p", POINT))
    rows = []
    for name, matrix in SCENES:
        area = determinant(matrix)
        if not math.isfinite(area):
            raise ValueError("行列式必须有限")
        for label, point in points:
            rows.append((name, label, apply(matrix, point), area))
    return rows


def main() -> None:
    """打印坐标表，供学习或与 C 输出对照。

    :return: 无。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", action="store_true", help="输出 C/Python 对照用 CSV")
    args = parser.parse_args()
    if args.csv:
        print("scene,point,x,y,determinant")
    for name, label, point, area in scene_rows():
        if args.csv:
            print("{},{},{:.17g},{:.17g},{:.17g}".format(name, label, point.x,
                                                            point.y, area))
        else:
            print("{:<20} {:<2} -> ({:5.1f}, {:5.1f}) det={:4.1f}".format(
                name, label, point.x, point.y, area))


if __name__ == "__main__":
    main()
