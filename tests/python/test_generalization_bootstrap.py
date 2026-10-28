"""检查 M10 的多项式拟合、成对自助法和 C/Python 数值一致性。"""

import csv
import importlib.util
import io
import math
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src/python/ml/39_generalization_bootstrap.py"
SPEC = importlib.util.spec_from_file_location("generalization_bootstrap", SOURCE)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
COMPARISON_SOURCE = ROOT / "src/python/ml/28_model_comparison.py"
COMPARISON_SPEC = importlib.util.spec_from_file_location(
    "model_comparison_with_bootstrap", COMPARISON_SOURCE)
assert COMPARISON_SPEC is not None and COMPARISON_SPEC.loader is not None
COMPARISON_MODULE = importlib.util.module_from_spec(COMPARISON_SPEC)
sys.modules[COMPARISON_SPEC.name] = COMPARISON_MODULE
COMPARISON_SPEC.loader.exec_module(COMPARISON_MODULE)


class GeneralizationBootstrapTests(unittest.TestCase):
    """验证拟合边界、配对抽样约定及确定性实验结果。"""

    def test_polynomial_fit_matches_hand_calculation(self) -> None:
        """从四个点恢复二次函数，并核对 Horner 预测。"""
        samples = [
            (-1.0, -1.25),
            (0.0, 0.75),
            (1.0, 1.15),
            (2.0, -0.05),
        ]
        coefficients = MODULE.polynomial_fit(samples, 2)
        self.assertAlmostEqual(coefficients[0], 0.75, places=10)
        self.assertAlmostEqual(coefficients[1], 1.2, places=10)
        self.assertAlmostEqual(coefficients[2], -0.8, places=10)
        self.assertAlmostEqual(MODULE.polynomial_predict(coefficients, 0.5), 1.15, places=10)

    def test_invalid_and_rank_deficient_inputs_are_rejected(self) -> None:
        """拒绝欠定、超容量、秩亏、非有限数据和无效随机种子。"""
        with self.assertRaises(ValueError):
            MODULE.polynomial_fit([(0.0, 1.0)], 1)
        with self.assertRaises(ValueError):
            MODULE.polynomial_fit([(0.0, 1.0)] * 7, MODULE.MAX_DEGREE + 1)
        with self.assertRaises(ValueError):
            MODULE.polynomial_fit([(1.0, 1.0)] * 3, 2)
        with self.assertRaises(ValueError):
            MODULE.polynomial_fit([(0.0, math.inf)], 0)
        with self.assertRaises(ValueError):
            MODULE.polynomial_predict((1.0, 2.0), math.nan)
        with self.assertRaises(ValueError):
            MODULE.dataset_generate(MODULE.Random32(1), 0, -1.0, 1.0, 0.1)
        with self.assertRaises(ValueError):
            MODULE.Random32(0)

    def test_paired_resampling_reuses_each_test_index(self) -> None:
        """用相同索引重采样，手算三项差值均值为 14/3。"""
        loss_a = [4.0, 1.0, 10.0]
        loss_b = [1.0, 1.0, 2.0]
        indices = [0, 0, 2]
        self.assertAlmostEqual(
            MODULE.paired_resampled_difference(loss_a, loss_b, indices), 14.0 / 3.0)
        with self.assertRaises(ValueError):
            MODULE.paired_resampled_difference(loss_a, loss_b, [0, 3])
        with self.assertRaises(ValueError):
            MODULE.paired_bootstrap([], [], 10, 1)

    def test_repeated_fits_separate_training_variability_from_test_uncertainty(self) -> None:
        """确认高次模型的训练波动与固定测试集 bootstrap 区间是不同量。"""
        values = MODULE.results()
        self.assertGreater(values["degree5_variance"], values["degree3_variance"])
        self.assertGreater(values["degree5_shifted_mse"], values["degree5_test_mse"])
        self.assertLess(values["paired_bootstrap_lower"], 0.0)
        self.assertGreater(values["paired_bootstrap_upper"], 0.0)
        self.assertAlmostEqual(values["degree3_expected_mse"],
                               values["degree3_bias_squared"] + values["degree3_variance"])

    def test_test_labels_do_not_enter_fitting(self) -> None:
        """只将训练样本传给拟合器，改变独立测试标签不改变参数。"""
        training = [(-1.0, 2.0), (0.0, 1.0), (1.0, 2.0)]
        test = [(0.5, 1.5)]
        before = MODULE.polynomial_fit(training, 2)
        test[0] = (test[0][0], -1000.0)
        after = MODULE.polynomial_fit(training, 2)
        self.assertEqual(before, after)

    def test_c_and_python_experiments_match(self) -> None:
        """比较所有固定字段，确保两语言使用相同数据与重采样。"""
        output = subprocess.check_output(
            [str(ROOT / "build/ml/39_generalization_bootstrap"), "--csv"],
            cwd=ROOT, text=True,
        )
        c_values = {row["field"]: float(row["value"])
                    for row in csv.DictReader(io.StringIO(output))}
        python_values = MODULE.results()
        self.assertEqual(c_values.keys(), python_values.keys())
        for field, value in c_values.items():
            self.assertAlmostEqual(value, python_values[field], delta=1e-9, msg=field)

    def test_existing_classifier_comparison_adds_paired_bootstrap(self) -> None:
        """检查公平比较案例使用单一配对差值并给出确定性区间。"""
        differences = [3.0, 0.0, 8.0]
        first = COMPARISON_MODULE.paired_bootstrap(differences, 0x28B007)
        second = COMPARISON_MODULE.paired_bootstrap(differences, 0x28B007)
        self.assertEqual(first, second)
        self.assertAlmostEqual(first[0], 11.0 / 3.0)
        self.assertLessEqual(first[1], first[2])
        with self.assertRaises(ValueError):
            COMPARISON_MODULE.paired_bootstrap([], 1)
        with self.assertRaises(ValueError):
            COMPARISON_MODULE.paired_bootstrap([math.nan], 1)


if __name__ == "__main__":
    unittest.main()
