"""检查 M07 的投影、分解、秩亏和 C/Python 对照。"""

import csv
import importlib.util
import io
import math
from pathlib import Path
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src/python/ml/36_projection_decomposition.py"
SPEC = importlib.util.spec_from_file_location("projection_decomposition", SOURCE)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
import sys
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class ProjectionDecompositionTests(unittest.TestCase):
    """使用固定手算数据验证三个数值路径。"""

    def test_qr_projection_and_residual(self) -> None:
        """检查最小二乘解、正交性和残差正交。"""
        a = ((1, 0), (1, 1), (1, 2))
        b = (1, 2, 2)
        result = MODULE.qr_solve(a, b)
        self.assertAlmostEqual(result.x[0], 7 / 6)
        self.assertAlmostEqual(result.x[1], 1 / 2)
        self.assertAlmostEqual(result.residual, math.sqrt(1 / 6))
        for column in range(2):
            self.assertAlmostEqual(sum(result.q[i][column] ** 2 for i in range(3)), 1)
            self.assertAlmostEqual(sum(result.q[i][column] *
                                       (b[i] - sum(a[i][j] * result.x[j]
                                                   for j in range(2)))
                                       for i in range(3)), 0)
        self.assertAlmostEqual(sum(row[0] * row[1] for row in result.q), 0)

    def test_svd_reconstruction(self) -> None:
        """检查两个奇异值与各格重建。"""
        a = ((2, 1), (1, 2))
        result = MODULE.svd2(a)
        self.assertEqual(result.sigma, (3, 1))
        for basis in (result.u, result.v):
            for column in range(2):
                self.assertAlmostEqual(sum(basis[row][column] ** 2 for row in range(2)), 1)
            self.assertAlmostEqual(sum(basis[row][0] * basis[row][1]
                                       for row in range(2)), 0)
        for i in range(2):
            for j in range(2):
                restored = sum(result.u[i][k] * result.sigma[k] * result.v[j][k]
                               for k in range(2))
                self.assertAlmostEqual(restored, a[i][j])
        self.assertLess(result.reconstruction_error, 1e-12)

    def test_rank_and_ill_conditioning(self) -> None:
        """检查秩亏状态，以及病态输入中两种求解的误差。"""
        with self.assertRaises(ValueError):
            MODULE.qr_solve(((1, 2), (2, 4), (3, 6)), (1, 2, 3))
        with self.assertRaises(ValueError):
            MODULE.qr_solve(((1, 0),), (1,))
        with self.assertRaises(ValueError):
            MODULE.svd2(((1, 2), (2, 4)))
        with self.assertRaises(ValueError):
            MODULE.svd2(((1, math.nan), (2, 3)))
        values = MODULE.results()
        self.assertLess(values["qr_error"], 1e-8)
        self.assertGreater(values["normal_error"], values["qr_error"] * 1000)

    def test_c_python_csv(self) -> None:
        """逐字段对照稳定量，并检查病态对照的误差数量级。"""
        output = subprocess.check_output(
            [str(ROOT / "build/ml/36_projection_decomposition"), "--csv"],
            cwd=ROOT, text=True,
        )
        c_values = {row["field"]: float(row["value"])
                    for row in csv.DictReader(io.StringIO(output))}
        python_values = MODULE.results()
        self.assertEqual(c_values.keys(), python_values.keys())
        for key in ("x0", "x1", "residual", "sigma0", "sigma1", "reconstruction"):
            self.assertAlmostEqual(c_values[key], python_values[key], delta=1e-12)
        self.assertLess(c_values["qr_error"], 1e-8)
        self.assertGreater(c_values["normal_error"], 1e-5)


if __name__ == "__main__":
    unittest.main()
