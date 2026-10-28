"""核对 M04 闭式估计、失败路径和 C/Python 固定实验。"""

from __future__ import annotations

import csv
import importlib.util
import math
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src/python/ml/33_probability_estimation.py"
spec = importlib.util.spec_from_file_location("probability_estimation", SOURCE)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


class ProbabilityEstimationTest(unittest.TestCase):
    """验证教学小数据的结果与边界约定。"""

    def test_coin_closed_form_and_boundary_counts(self) -> None:
        """核对 MLE、后验均值、MAP 和全正/全反计数。"""
        result = module.estimate_coin(7, 3, 2, 2)
        self.assertAlmostEqual(result.mle, 0.7)
        self.assertAlmostEqual(result.posterior_mean, 9 / 14)
        self.assertAlmostEqual(result.map, 2 / 3)
        self.assertEqual(result.posterior_alpha, 9)
        self.assertEqual(result.posterior_beta, 5)
        self.assertEqual(module.estimate_coin(0, 10, 2, 2).mle, 0)
        self.assertEqual(module.estimate_coin(10, 0, 2, 2).mle, 1)
        with self.assertRaises(ValueError):
            module.estimate_coin(0, 0, 2, 2)
        with self.assertRaises(ValueError):
            module.estimate_coin(7, 3, 1, 2)
        with self.assertRaises(ValueError):
            module.coin_mean_nll(1, 0, 0)
        self.assertEqual(module.coin_mean_nll(0, 3, 0), 0)

    def test_gaussian_and_degenerate_data(self) -> None:
        """核对两个方差分母和退化模型的密度失败。"""
        result = module.estimate_gaussian((1, 2, 3, 4))
        self.assertEqual(result.mean, 2.5)
        self.assertEqual(result.mle_variance, 1.25)
        self.assertAlmostEqual(result.unbiased_variance, 5 / 3)
        self.assertAlmostEqual(module.gaussian_density(result, 2.5),
                               1 / math.sqrt(2 * math.pi * 1.25))
        self.assertIsNone(module.estimate_gaussian((4,)).unbiased_variance)
        with self.assertRaises(ValueError):
            module.estimate_gaussian(())
        with self.assertRaises(ValueError):
            module.estimate_gaussian((1, float("nan")))
        with self.assertRaises(ValueError):
            module.gaussian_density(module.estimate_gaussian((4, 4, 4)), 4)

    def test_c_python_csv(self) -> None:
        """比较相同训练和独立测试数据的全部输出字段。"""
        c_output = subprocess.check_output(
            [str(ROOT / "build/ml/33_probability_estimation"), "--csv"], text=True,
            cwd=ROOT,
        )
        py_output = subprocess.check_output([sys.executable, str(SOURCE), "--csv"],
                                            text=True, cwd=ROOT)
        c_rows = dict(csv.reader(c_output.splitlines()[1:]))
        py_rows = dict(csv.reader(py_output.splitlines()[1:]))
        self.assertEqual(c_rows.keys(), py_rows.keys())
        for field in c_rows:
            self.assertAlmostEqual(float(c_rows[field]), float(py_rows[field]), places=12)
        self.assertLess(float(c_rows["coin_test_nll_map"]),
                        float(c_rows["coin_test_nll_baseline"]))


if __name__ == "__main__":
    unittest.main()
