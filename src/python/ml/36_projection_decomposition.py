"""用固定小矩阵观察 QR 投影、最小二乘及二维奇异值分解。"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class QrResult:
    """保存两列正交向量、上三角矩阵、最小二乘解及残差范数。"""

    q: tuple[tuple[float, float], ...]
    r: tuple[tuple[float, float], tuple[float, float]]
    x: tuple[float, float]
    residual: float


@dataclass(frozen=True)
class SvdResult:
    """保存二维满秩矩阵的左右方向、奇异值及重建误差。"""

    u: tuple[tuple[float, float], tuple[float, float]]
    v: tuple[tuple[float, float], tuple[float, float]]
    sigma: tuple[float, float]
    reconstruction_error: float


def _check_data(a: tuple[tuple[float, float], ...], b: tuple[float, ...]) -> None:
    """检查本例固定两列矩阵的形状与有限性。

    :param a: 两列且有 2 至 4 行的矩阵。
    :param b: 与矩阵同样行数的目标向量。
    :return: 无。
    :raises ValueError: 形状或元素无效时抛出。
    """
    if not 2 <= len(a) <= 4 or len(b) != len(a):
        raise ValueError("需要 2 至 4 行及相同长度的目标")
    if any(len(row) != 2 or any(not math.isfinite(v) for v in row) for row in a):
        raise ValueError("矩阵必须为有限的两列")
    if any(not math.isfinite(v) for v in b):
        raise ValueError("目标必须有限")


def qr_solve(a: tuple[tuple[float, float], ...], b: tuple[float, ...]) -> QrResult:
    """两遍 Gram-Schmidt 正交化后求满列秩最小二乘解。

    :param a: 2 至 4 行、两列的有限矩阵。
    :param b: 与矩阵同样行数的有限目标。
    :return: 正交分解、解和残差范数。
    :raises ValueError: 形状、秩或数值无效时抛出。
    """
    _check_data(a, b)
    norm0 = 0.0
    norm1 = 0.0
    for row in a:
        norm0 = math.hypot(norm0, row[0])
        norm1 = math.hypot(norm1, row[1])
    if not math.isfinite(norm0) or not math.isfinite(norm1) or norm0 == 0 or norm1 == 0:
        raise ValueError("零列或非有限范数")
    q0 = [row[0] / norm0 for row in a]
    r01 = sum(q0[i] * a[i][1] for i in range(len(a)))
    q1 = [a[i][1] - r01 * q0[i] for i in range(len(a))]
    correction = sum(q0[i] * q1[i] for i in range(len(a)))
    r01 += correction
    q1 = [q1[i] - correction * q0[i] for i in range(len(a))]
    r11 = 0.0
    for value in q1:
        r11 = math.hypot(r11, value)
    if not math.isfinite(r11) or r11 <= 1e-12 * norm1:
        raise ValueError("两列秩亏或数值上不可区分")
    q1 = [value / r11 for value in q1]
    projected0 = sum(q0[i] * b[i] for i in range(len(a)))
    projected1 = sum(q1[i] * b[i] for i in range(len(a)))
    x1 = projected1 / r11
    x0 = (projected0 - r01 * x1) / norm0
    residual = 0.0
    for i, row in enumerate(a):
        error = b[i] - row[0] * x0 - row[1] * x1
        residual = math.hypot(residual, error)
    if any(not math.isfinite(value) for value in (x0, x1, residual)):
        raise ValueError("求解结果非有限")
    q = tuple((q0[i], q1[i]) for i in range(len(a)))
    return QrResult(q, ((norm0, r01), (0.0, r11)), (x0, x1), residual)


def normal_solve(a: tuple[tuple[float, float], ...], b: tuple[float, ...]) -> tuple[float, float]:
    """以正规方程作为病态数据对照。

    :param a: 2 至 4 行、两列的有限矩阵。
    :param b: 与矩阵同样行数的有限目标。
    :return: 两个参数；仅用于固定数据比较。
    :raises ValueError: 形状、行列式或结果无效时抛出。
    """
    _check_data(a, b)
    s00 = s01 = s11 = t0 = t1 = 0.0
    for i, row in enumerate(a):
        s00 += row[0] * row[0]
        s01 += row[0] * row[1]
        s11 += row[1] * row[1]
        t0 += row[0] * b[i]
        t1 += row[1] * b[i]
    determinant = s00 * s11 - s01 * s01
    if not math.isfinite(determinant) or determinant <= 0:
        raise ValueError("正规方程不可解")
    x = ((t0 * s11 - t1 * s01) / determinant,
         (s00 * t1 - s01 * t0) / determinant)
    if any(not math.isfinite(v) for v in x):
        raise ValueError("正规方程结果非有限")
    return x


def svd2(a: tuple[tuple[float, float], tuple[float, float]]) -> SvdResult:
    """由转置乘积的右方向构造二维满秩奇异值分解。

    :param a: 有限的 2×2 满秩矩阵。
    :return: 左右正交方向、奇异值与 Frobenius 重建误差。
    :raises ValueError: 形状、秩、正交性或重建无效时抛出。
    """
    if len(a) != 2 or any(len(row) != 2 or any(not math.isfinite(v) for v in row)
                          for row in a):
        raise ValueError("需要有限的 2×2 矩阵")
    g00 = a[0][0] ** 2 + a[1][0] ** 2
    g01 = a[0][0] * a[0][1] + a[1][0] * a[1][1]
    g11 = a[0][1] ** 2 + a[1][1] ** 2
    angle = 0.5 * math.atan2(2 * g01, g00 - g11)
    v = ((math.cos(angle), -math.sin(angle)),
         (math.sin(angle), math.cos(angle)))
    sigma = []
    columns = []
    for k in range(2):
        y0 = a[0][0] * v[0][k] + a[0][1] * v[1][k]
        y1 = a[1][0] * v[0][k] + a[1][1] * v[1][k]
        length = math.hypot(y0, y1)
        if not math.isfinite(length) or length == 0:
            raise ValueError("矩阵秩亏")
        sigma.append(length)
        columns.append((y0 / length, y1 / length))
    if sigma[1] <= 1e-12 * sigma[0]:
        raise ValueError("矩阵秩亏或数值上不可区分")
    u = ((columns[0][0], columns[1][0]),
         (columns[0][1], columns[1][1]))
    dot = sum(columns[0][i] * columns[1][i] for i in range(2))
    if not math.isfinite(dot) or abs(dot) > 1e-10 or sigma[0] < sigma[1]:
        raise ValueError("奇异方向失去正交性")
    error = 0.0
    for i in range(2):
        for j in range(2):
            restored = sum(u[i][k] * sigma[k] * v[j][k] for k in range(2))
            error = math.hypot(error, restored - a[i][j])
    if not math.isfinite(error) or error > 1e-9:
        raise ValueError("矩阵重建误差过大")
    return SvdResult(u, v, (sigma[0], sigma[1]), error)


def results() -> dict[str, float]:
    """计算固定例子的输出字段。

    :return: QR 解、残差、奇异值、重建与病态对照误差。
    """
    qr = qr_solve(((1, 0), (1, 1), (1, 2)), (1, 2, 2))
    svd = svd2(((2, 1), (1, 2)))
    h = 1e-6
    a = ((1., 1.), (1., 1. + h), (1., 1. - h))
    b = (2., 2. + h, 2. - h)
    difficult_qr = qr_solve(a, b).x
    normal = normal_solve(a, b)
    return {
        "x0": qr.x[0], "x1": qr.x[1], "residual": qr.residual,
        "sigma0": svd.sigma[0], "sigma1": svd.sigma[1],
        "reconstruction": svd.reconstruction_error,
        "qr_error": max(abs(value - 1) for value in difficult_qr),
        "normal_error": max(abs(value - 1) for value in normal),
    }


def main() -> None:
    """运行固定实验或输出跨语言 CSV。

    :return: 无。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", action="store_true", help="输出 field,value")
    args = parser.parse_args()
    values = results()
    if args.csv:
        print("field,value")
        for field, value in values.items():
            print(f"{field},{value:.17g}")
        return
    print("QR 最小二乘：x=({:.8f}, {:.8f})，残差范数={:.8f}".format(
        values["x0"], values["x1"], values["residual"]))
    print("二维 SVD：奇异值=({:.8f}, {:.8f})，重建误差={:.3g}".format(
        values["sigma0"], values["sigma1"], values["reconstruction"]))
    print("近共线输入：QR 参数误差={:.6g}，正规方程参数误差={:.6g}".format(
        values["qr_error"], values["normal_error"]))


if __name__ == "__main__":
    main()
