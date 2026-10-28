"""验证 D01 的归一化梯度、运行统计、初始化和 C/Python 数值一致性。"""

import csv
import importlib.util
import io
import math
from pathlib import Path
import subprocess
import sys
from typing import Callable
import unittest

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src/python/ml/41_normalization_residual.py"
SPEC = importlib.util.spec_from_file_location("normalization_residual", SOURCE)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _objective(matrix: tuple[tuple[float, ...], ...]) -> float:
    """计算固定输出与上游梯度的内积。

    :param matrix: 与 D01 固定上游梯度同形状的输出。
    :return: 标量目标值。
    """
    return sum(
        matrix[row][feature] * MODULE.UPSTREAM[row][feature]
        for row in range(len(matrix))
        for feature in range(len(matrix[row]))
    )


def _finite_difference(
    evaluate: Callable[[list[float]], float],
    values: list[float],
    index: int,
    step: float = 1e-6,
) -> float:
    """对一组扁平参数计算中心差分。

    :param evaluate: 接收参数列表并返回标量目标的函数。
    :param values: 原始参数列表。
    :param index: 要扰动的元素下标。
    :param step: 中心差分步长。
    :return: 数值梯度近似。
    """
    high = list(values)
    low = list(values)
    high[index] += step
    low[index] -= step
    return (evaluate(high) - evaluate(low)) / (2.0 * step)


class NormalizationResidualTests(unittest.TestCase):
    """用小型确定性数组核对推导公式和实现边界。"""

    def test_hand_calculated_forward_values(self) -> None:
        """核对 LN、BN 和 RMSNorm 第一项的手算值。"""
        layer, _ = MODULE.layer_norm_forward(
            MODULE.INPUT, MODULE.GAMMA, MODULE.BETA)
        batch, _, _ = MODULE.batch_norm_forward(
            MODULE.INPUT, MODULE.GAMMA, MODULE.BETA,
            MODULE.batch_norm_state_init(3), True)
        rms, _ = MODULE.rms_norm_forward(MODULE.INPUT, MODULE.GAMMA)
        expected_layer = MODULE.GAMMA[0] * (-1.0 / math.sqrt(2.0 / 3.0 + 0.01)) + MODULE.BETA[0]
        expected_batch = MODULE.GAMMA[0] * (-0.5 / math.sqrt(0.25 + 0.01)) + MODULE.BETA[0]
        expected_rms = MODULE.GAMMA[0] / math.sqrt(14.0 / 3.0 + 0.01)
        self.assertAlmostEqual(layer[0][0], expected_layer, places=12)
        self.assertAlmostEqual(batch[0][0], expected_batch, places=12)
        self.assertAlmostEqual(rms[0][0], expected_rms, places=12)

    def test_layer_norm_gradients_by_central_difference(self) -> None:
        """检查 LN 对输入、gamma 和 beta 的全部解析梯度。"""
        layer, cache = MODULE.layer_norm_forward(MODULE.INPUT, MODULE.GAMMA, MODULE.BETA)
        del layer
        input_gradient, gamma_gradient, beta_gradient = MODULE.layer_norm_backward(
            cache, MODULE.GAMMA, MODULE.UPSTREAM)
        groups = (
            ([value for row in MODULE.INPUT for value in row],
             [value for row in input_gradient for value in row], 0),
            (list(MODULE.GAMMA), list(gamma_gradient), 1),
            (list(MODULE.BETA), list(beta_gradient), 2),
        )

        def evaluate(parameters: list[float], group: int) -> float:
            inputs = [list(row) for row in MODULE.INPUT]
            gamma = list(MODULE.GAMMA)
            beta = list(MODULE.BETA)
            if group == 0:
                inputs = [parameters[row * 3:(row + 1) * 3] for row in range(2)]
            elif group == 1:
                gamma = parameters
            else:
                beta = parameters
            output, _ = MODULE.layer_norm_forward(inputs, gamma, beta)
            return _objective(output)

        for values, analytical, group in groups:
            for index, expected in enumerate(analytical):
                numerical = _finite_difference(
                    lambda perturbed: evaluate(perturbed, group), values, index)
                self.assertAlmostEqual(numerical, expected, delta=2e-7)

    def test_batch_norm_gradients_by_central_difference(self) -> None:
        """检查训练态 BN 对输入、gamma 和 beta 的全部解析梯度。"""
        _, cache, _ = MODULE.batch_norm_forward(
            MODULE.INPUT, MODULE.GAMMA, MODULE.BETA,
            MODULE.batch_norm_state_init(3), True)
        input_gradient, gamma_gradient, beta_gradient = MODULE.batch_norm_backward(
            cache, MODULE.GAMMA, MODULE.UPSTREAM)
        groups = (
            ([value for row in MODULE.INPUT for value in row],
             [value for row in input_gradient for value in row], 0),
            (list(MODULE.GAMMA), list(gamma_gradient), 1),
            (list(MODULE.BETA), list(beta_gradient), 2),
        )

        def evaluate(parameters: list[float], group: int) -> float:
            inputs = [list(row) for row in MODULE.INPUT]
            gamma = list(MODULE.GAMMA)
            beta = list(MODULE.BETA)
            if group == 0:
                inputs = [parameters[row * 3:(row + 1) * 3] for row in range(2)]
            elif group == 1:
                gamma = parameters
            else:
                beta = parameters
            output, _, _ = MODULE.batch_norm_forward(
                inputs, gamma, beta, MODULE.batch_norm_state_init(3), True)
            return _objective(output)

        for values, analytical, group in groups:
            for index, expected in enumerate(analytical):
                numerical = _finite_difference(
                    lambda perturbed: evaluate(perturbed, group), values, index)
                self.assertAlmostEqual(numerical, expected, delta=2e-7)

    def test_rms_norm_gradients_by_central_difference(self) -> None:
        """检查 RMSNorm 对输入和 gamma 的全部解析梯度。"""
        _, cache = MODULE.rms_norm_forward(MODULE.INPUT, MODULE.GAMMA)
        input_gradient, gamma_gradient = MODULE.rms_norm_backward(
            cache, MODULE.GAMMA, MODULE.UPSTREAM)
        groups = (
            ([value for row in MODULE.INPUT for value in row],
             [value for row in input_gradient for value in row], 0),
            (list(MODULE.GAMMA), list(gamma_gradient), 1),
        )

        def evaluate(parameters: list[float], group: int) -> float:
            inputs = [list(row) for row in MODULE.INPUT]
            gamma = list(MODULE.GAMMA)
            if group == 0:
                inputs = [parameters[row * 3:(row + 1) * 3] for row in range(2)]
            else:
                gamma = parameters
            output, _ = MODULE.rms_norm_forward(inputs, gamma)
            return _objective(output)

        for values, analytical, group in groups:
            for index, expected in enumerate(analytical):
                numerical = _finite_difference(
                    lambda perturbed: evaluate(perturbed, group), values, index)
                self.assertAlmostEqual(numerical, expected, delta=2e-7)

    def test_batch_norm_uses_saved_statistics_for_held_out_inference(self) -> None:
        """验证训练态更新运行统计、推理态复用且不改写状态。"""
        training_state = MODULE.batch_norm_state_init(3)
        _, train_cache, fitted_state = MODULE.batch_norm_forward(
            MODULE.INPUT, MODULE.GAMMA, MODULE.BETA, training_state, True)
        self.assertTrue(train_cache.training)
        for actual, expected in zip(fitted_state.running_mean, (0.15, 0.3, 0.45)):
            self.assertAlmostEqual(actual, expected, places=14)
        for actual, expected in zip(fitted_state.running_variance, (0.925, 1.0, 1.125)):
            self.assertAlmostEqual(actual, expected, places=14)
        state_before_inference = fitted_state
        held_out = ((3.0, 6.0, 9.0),)
        output, eval_cache, reused_state = MODULE.batch_norm_forward(
            held_out, MODULE.GAMMA, MODULE.BETA, fitted_state, False)
        self.assertFalse(eval_cache.training)
        self.assertIs(reused_state, fitted_state)
        self.assertEqual(fitted_state, state_before_inference)
        expected = MODULE.GAMMA[0] * (3.0 - 0.15) / math.sqrt(0.925 + 0.01) + MODULE.BETA[0]
        self.assertAlmostEqual(output[0][0], expected, places=12)
        dx, _, _ = MODULE.batch_norm_backward(
            eval_cache, MODULE.GAMMA, ((1.0, 1.0, 1.0),))
        self.assertAlmostEqual(dx[0][0], MODULE.GAMMA[0] / math.sqrt(0.925 + 0.01))

    def test_singleton_and_constant_inputs_stay_finite(self) -> None:
        """验证 BN 单样本退化及 LN/RMSNorm 零方差行为。"""
        gamma = (2.0, -3.0)
        beta = (0.25, -0.75)
        output, cache, _ = MODULE.batch_norm_forward(
            ((5.0, -2.0),), gamma, beta,
            MODULE.batch_norm_state_init(2), True, epsilon=1e-5)
        self.assertEqual(output, ((beta[0], beta[1]),))
        dx, dgamma, _ = MODULE.batch_norm_backward(cache, gamma, ((5.0, -2.0),))
        self.assertEqual(dx, ((0.0, 0.0),))
        self.assertEqual(dgamma, (0.0, 0.0))
        constant = ((4.0, 4.0, 4.0),)
        layer, _ = MODULE.layer_norm_forward(constant, (1.0, 1.0, 1.0), (0.0, 0.0, 0.0))
        rms, _ = MODULE.rms_norm_forward(constant, (1.0, 1.0, 1.0))
        self.assertTrue(all(math.isfinite(value) for value in layer[0] + rms[0]))

    def test_invalid_inputs_initialization_and_residual_path(self) -> None:
        """检查形状、非有限值、初始化边界及残差传梯度结果。"""
        with self.assertRaises(ValueError):
            MODULE.layer_norm_forward(((1.0, 2.0), (3.0,)), (1.0, 1.0), (0.0, 0.0))
        with self.assertRaises(ValueError):
            MODULE.rms_norm_forward(((math.nan,),), (1.0,))
        with self.assertRaises(ValueError):
            MODULE.layer_norm_forward(((1e308, -1e308),), (0.0, 0.0), (0.0, 0.0))
        with self.assertRaises(ValueError):
            MODULE.batch_norm_forward(
                MODULE.INPUT, MODULE.GAMMA, MODULE.BETA,
                MODULE.batch_norm_state_init(3), True, momentum=1.0)
        untouched_state = MODULE.batch_norm_state_init(3)
        with self.assertRaises(ValueError):
            MODULE.batch_norm_forward(
                ((math.nan, 2.0, 3.0),), MODULE.GAMMA, MODULE.BETA,
                untouched_state, True)
        self.assertEqual(untouched_state, MODULE.batch_norm_state_init(3))
        self.assertAlmostEqual(MODULE.initializer_variance(4, 3, "he"), 0.5)
        self.assertAlmostEqual(MODULE.initializer_variance(4, 3, "glorot"), 2.0 / 7.0)
        weights, variance = MODULE.initialize_weights(12, 4, 3, "he", 41)
        self.assertEqual(weights, MODULE.initialize_weights(12, 4, 3, "he", 41)[0])
        self.assertTrue(math.isfinite(variance))
        with self.assertRaises(ValueError):
            MODULE.initializer_variance(0, 3, "he")
        with self.assertRaises(ValueError):
            MODULE.initialize_weights(13, 4, 3, "he", 41)
        plain_value, plain_gradient = MODULE.gradient_chain(8, -0.1, False)
        residual_value, residual_gradient = MODULE.gradient_chain(8, -0.1, True)
        self.assertAlmostEqual(plain_gradient, (-0.1) ** 8)
        self.assertAlmostEqual(residual_gradient, 0.9 ** 8)
        self.assertEqual(plain_value, plain_gradient)
        self.assertEqual(residual_value, residual_gradient)
        step = 1e-6
        point = 0.37
        numerical = (
            residual_value * (point + step) - residual_value * (point - step)
        ) / (2.0 * step)
        self.assertAlmostEqual(numerical, residual_gradient, delta=1e-10)
        with self.assertRaises(ValueError):
            MODULE.gradient_chain(0, -0.1, True)

    def test_c_and_python_share_the_same_numeric_fixture(self) -> None:
        """逐字段比较 C 与 Python 的归一化、初始化及残差数值。"""
        output = subprocess.check_output(
            [str(ROOT / "build/ml/41_normalization_residual"), "--csv"],
            cwd=ROOT,
            text=True,
        )
        c_values = {
            row["field"]: float(row["value"])
            for row in csv.DictReader(io.StringIO(output))
        }
        python_values = MODULE.results()
        self.assertEqual(c_values.keys(), python_values.keys())
        for field, value in c_values.items():
            self.assertAlmostEqual(value, python_values[field], delta=2e-12, msg=field)


if __name__ == "__main__":
    unittest.main()
