"""验证 M06 大分数、差分误差、病态矩阵和 C/Python 对照。"""

from __future__ import annotations

import csv
import importlib.util
import math
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src/python/ml/35_numerical_stability.py"
spec = importlib.util.spec_from_file_location("numerical_stability", SOURCE)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


class NumericalStabilityTest(unittest.TestCase):
    """对小型确定性实验做公式及失败路径验证。"""

    def test_large_logits(self) -> None:
        """朴素指数溢出，减最大值后概率有限且保持平移不变。"""
        self.assertIsNone(module.naive_softmax((1000, 1001, 1002)))
        baseline = module.naive_softmax((0, 1, 2))
        probabilities, logsumexp = module.stable_softmax((1000, 1001, 1002))
        assert baseline is not None
        for actual, expected in zip(probabilities, baseline):
            self.assertAlmostEqual(actual, expected, places=12)
        self.assertAlmostEqual(sum(probabilities), 1, places=12)
        self.assertAlmostEqual(logsumexp, 1002 + math.log(1 + math.exp(-1) + math.exp(-2)))
        with self.assertRaises(ValueError):
            module.stable_softmax((0, float("nan"), 2))

    def test_difference_step_scan(self) -> None:
        """大步长呈截断误差，小步长呈舍入放大。"""
        errors = [abs(module.cubic_difference(step) - 3) for step in module.STEPS]
        self.assertAlmostEqual(errors[0], 0.01, places=12)
        self.assertLess(errors[3], 1e-8)
        self.assertGreater(errors[-1], errors[3])
        with self.assertRaises(ValueError):
            module.cubic_difference(0)

    def test_near_singular_system(self) -> None:
        """检查解析解、扰动放大和条件数上界。"""
        result = module.near_singular_system(1e-8, 1e-10)
        self.assertAlmostEqual(result.base_x[0], 1, places=12)
        self.assertAlmostEqual(result.base_x[1], 1, places=12)
        self.assertAlmostEqual(result.changed_x[0], 0.99, places=6)
        self.assertAlmostEqual(result.changed_x[1], 1.01, places=6)
        self.assertGreater(result.relative_amplification, 1e8)
        self.assertLess(result.relative_amplification, result.condition_infinity)
        with self.assertRaises(ValueError):
            module.near_singular_system(1e-20, 1e-10)
        with self.assertRaises(ValueError):
            module.near_singular_system(1e-8, 1e-20)

    def test_c_python_csv(self) -> None:
        """比较全部固定实验字段，不依赖两语言随机数。"""
        c_output = subprocess.check_output([str(ROOT / "build/ml/35_numerical_stability"),
                                            "--csv"], text=True, cwd=ROOT)
        py_output = subprocess.check_output([sys.executable, str(SOURCE), "--csv"],
                                            text=True, cwd=ROOT)
        c_rows = dict(csv.reader(c_output.splitlines()[1:]))
        py_rows = dict(csv.reader(py_output.splitlines()[1:]))
        self.assertEqual(c_rows.keys(), py_rows.keys())
        for field in c_rows:
            self.assertAlmostEqual(float(c_rows[field]), float(py_rows[field]), places=11)


if __name__ == "__main__":
    unittest.main()
