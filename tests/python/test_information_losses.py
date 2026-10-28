"""验证 M05 损失手算、梯度、无穷边界及两语言一致性。"""

from __future__ import annotations

import csv
import importlib.util
import math
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src/python/ml/34_information_losses.py"
spec = importlib.util.spec_from_file_location("information_losses", SOURCE)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


class InformationLossesTest(unittest.TestCase):
    """检查二类、三类及高斯损失的教学数值。"""

    def test_entropy_cross_entropy_and_kl(self) -> None:
        """核对手算值、相同分布 KL 零值及三个置信场景。"""
        soft = module.distribution_metrics((0.5, 0.5), (0.75, 0.25))
        self.assertAlmostEqual(soft.entropy, math.log(2))
        self.assertAlmostEqual(soft.cross_entropy, -0.5 * math.log(0.75 * 0.25))
        self.assertAlmostEqual(soft.kl, 0.5 * math.log(4 / 3))
        self.assertAlmostEqual(module.distribution_metrics((0.5, 0.5),
                                                           (0.5, 0.5)).kl, 0)
        values = module.results()
        self.assertLess(values["binary_correct_loss"], values["binary_uncertain_loss"])
        self.assertLess(values["binary_uncertain_loss"], values["binary_wrong_loss"])
        self.assertLess(values["ternary_correct_loss"], values["ternary_uncertain_loss"])
        self.assertLess(values["ternary_uncertain_loss"], values["ternary_wrong_loss"])

    def test_zero_probability_and_invalid_distribution(self) -> None:
        """零概率不静默裁剪；无效概率和零方差显式失败。"""
        impossible = module.distribution_metrics((0, 1), (1, 0))
        self.assertTrue(math.isinf(impossible.cross_entropy))
        self.assertTrue(math.isinf(impossible.kl))
        self.assertIsNone(impossible.logit_gradient)
        exact = module.distribution_metrics((1, 0), (1, 0))
        self.assertEqual(exact.cross_entropy, 0)
        self.assertEqual(exact.kl, 0)
        self.assertIsNone(exact.logit_gradient)
        for invalid in ((0.8, 0.8), (-0.1, 1.1), (float("nan"), 1)):
            with self.assertRaises(ValueError):
                module.distribution_metrics((0, 1), invalid)
        with self.assertRaises(ValueError):
            module.distribution_metrics((0, 1), (0.2, 0.3, 0.5))
        with self.assertRaises(ValueError):
            module.gaussian_loss(2, 1, 0)

    def test_gradients_against_centered_difference(self) -> None:
        """比较二类和三类 logit 梯度与中心差分。"""
        step = 1e-5
        for logits, target in (((0.0, math.log(3)), (0.0, 1.0)),
                               ((0.0, math.log(2), 0.0), (0.0, 1.0, 0.0))):
            def loss_at(changed: tuple[float, ...]) -> float:
                """将固定 logit 转成概率并计算交叉熵。

                :param changed: 本次待计算的类别分数。
                :return: 目标交叉熵。
                """
                exponential = tuple(math.exp(value) for value in changed)
                total = sum(exponential)
                predicted = tuple(value / total for value in exponential)
                return module.distribution_metrics(target, predicted).cross_entropy

            exponential = tuple(math.exp(value) for value in logits)
            total = sum(exponential)
            probabilities = tuple(value / total for value in exponential)
            analytical = module.distribution_metrics(target, probabilities).logit_gradient
            assert analytical is not None
            for index in range(len(logits)):
                high = list(logits)
                low = list(logits)
                high[index] += step
                low[index] -= step
                numerical = (loss_at(tuple(high)) - loss_at(tuple(low))) / (2 * step)
                self.assertLess(abs(analytical[index] - numerical), 1e-9)
        gaussian, gradient = module.gaussian_loss(2, 1, 1)
        self.assertAlmostEqual(gaussian, 0.5 * math.log(2 * math.pi) + 0.5)
        high = module.gaussian_loss(2, 1 + step, 1)[0]
        low = module.gaussian_loss(2, 1 - step, 1)[0]
        self.assertLess(abs(gradient - (high - low) / (2 * step)), 1e-9)

    def test_c_python_csv(self) -> None:
        """比较两语言在所有固定场景中的数值字段。"""
        c_output = subprocess.check_output([str(ROOT / "build/ml/34_information_losses"),
                                            "--csv"], text=True, cwd=ROOT)
        py_output = subprocess.check_output([sys.executable, str(SOURCE), "--csv"],
                                            text=True, cwd=ROOT)
        c_rows = dict(csv.reader(c_output.splitlines()[1:]))
        py_rows = dict(csv.reader(py_output.splitlines()[1:]))
        self.assertEqual(c_rows.keys(), py_rows.keys())
        for field in c_rows:
            self.assertAlmostEqual(float(c_rows[field]), float(py_rows[field]), places=12)


if __name__ == "__main__":
    unittest.main()
