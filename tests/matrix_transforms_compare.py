"""核对二维变换的 C/Python 中间值和几何边界。"""

from __future__ import annotations

import csv
import importlib.util
import io
import math
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
PYTHON_SOURCE = ROOT / "src/python/ml/30_matrix_transforms.py"
C_PROGRAM = ROOT / "build/ml/30_matrix_transforms"
spec = importlib.util.spec_from_file_location("matrix_transforms", PYTHON_SOURCE)
if spec is None or spec.loader is None:
    raise RuntimeError("无法加载 Python 对照示例")
matrix_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(matrix_module)


class MatrixTransformTests(unittest.TestCase):
    """比较两个程序的固定输入及失败边界。"""

    def test_c_python_rows(self) -> None:
        """逐行比较基向量、教学点和行列式。

        :return: 无。
        """
        c_output = subprocess.run([str(C_PROGRAM), "--csv"], check=True,
                                  capture_output=True, text=True, cwd=ROOT)
        py_output = subprocess.run([sys.executable, str(PYTHON_SOURCE), "--csv"],
                                   check=True, capture_output=True, text=True, cwd=ROOT)
        c_rows = list(csv.DictReader(io.StringIO(c_output.stdout)))
        py_rows = list(csv.DictReader(io.StringIO(py_output.stdout)))
        self.assertEqual(len(c_rows), 21)
        self.assertEqual(len(c_rows), len(py_rows))
        for c_row, py_row in zip(c_rows, py_rows):
            self.assertEqual((c_row["scene"], c_row["point"]),
                             (py_row["scene"], py_row["point"]))
            for field in ("x", "y", "determinant"):
                self.assertTrue(math.isclose(float(c_row[field]), float(py_row[field]),
                                              rel_tol=1e-12, abs_tol=1e-12),
                                (c_row, py_row, field))

    def test_geometry_and_order(self) -> None:
        """核对手算坐标、长度、秩亏与矩阵复合。

        :return: 无。
        """
        values = {name: matrix for name, matrix in matrix_module.SCENES}
        point = matrix_module.POINT
        self.assertEqual(matrix_module.apply(values["identity"], point), point)
        self.assertEqual(matrix_module.apply(values["scale"], point), (2.0, 1.0))
        rotated = matrix_module.apply(values["rotate_ccw_90"], point)
        self.assertEqual(rotated, (-2.0, 1.0))
        self.assertEqual(rotated.x ** 2 + rotated.y ** 2, 5.0)
        self.assertEqual(matrix_module.apply(values["shear_x"], point), (3.0, 2.0))
        self.assertEqual(matrix_module.apply(values["project_x"], point), (1.0, 0.0))
        self.assertEqual(matrix_module.determinant(values["project_x"]), 0.0)
        rs = matrix_module.multiply(values["rotate_ccw_90"], values["scale"])
        sr = matrix_module.multiply(values["scale"], values["rotate_ccw_90"])
        self.assertEqual(rs, values["rotate_after_scale"])
        self.assertEqual(sr, values["scale_after_rotate"])
        self.assertEqual(matrix_module.apply(rs, point), (-1.0, 2.0))
        self.assertEqual(matrix_module.apply(sr, point), (-4.0, 0.5))

    def test_nonfinite_rejected(self) -> None:
        """非法输入和溢出结果应显式失败。

        :return: 无。
        """
        identity = matrix_module.SCENES[0][1]
        with self.assertRaises(ValueError):
            matrix_module.apply(identity, matrix_module.Vec2(math.nan, 0.0))
        with self.assertRaises(ValueError):
            matrix_module.apply(matrix_module.Mat2((1e308, 1e308), (0.0, 1.0)),
                                matrix_module.POINT)
        with self.assertRaises(ValueError):
            matrix_module.multiply(matrix_module.Mat2((math.inf, 0.0), (0.0, 1.0)),
                                   identity)


if __name__ == "__main__":
    unittest.main()
