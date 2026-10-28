"""检查 M09 优化器轨迹、MLP 梯度和跨语言固定实验。"""

import csv
import importlib.util
import io
import math
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src/python/ml/38_optimization_diagnostics.py"
SPEC = importlib.util.spec_from_file_location("optimization_diagnostics", SOURCE)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class OptimizationDiagnosticsTests(unittest.TestCase):
    """使用同一初值、目标和更新预算验证优化器行为。"""

    def test_first_two_hand_steps(self) -> None:
        """核对 SGD、动量和 Adam 的前两次目标损失。"""
        values = MODULE.results()
        self.assertEqual(values["sgd_0"], 5.0)
        self.assertAlmostEqual(values["sgd_1"], 0.45)
        self.assertAlmostEqual(values["momentum_1"], 0.45)
        self.assertAlmostEqual(values["momentum_2"], 1.16)
        self.assertAlmostEqual(values["adam_1"], 4.05, delta=1e-8)
        self.assertAlmostEqual(values["adam_2"], 3.2032986743, delta=1e-9)

        point = (1.0, 1.0)
        state = MODULE.OptimizerState()
        _, gradient = MODULE.quadratic_evaluate(point)
        point, state = MODULE.optimizer_step(point, gradient, "adam", 0.1, state)
        self.assertAlmostEqual(point[0], 0.9, places=8)
        self.assertAlmostEqual(point[1], 0.9, places=8)
        _, gradient = MODULE.quadratic_evaluate(point)
        point, _ = MODULE.optimizer_step(point, gradient, "adam", 0.1, state)
        self.assertAlmostEqual(point[0], 0.80041223, places=8)
        self.assertAlmostEqual(point[1], 0.80041223, places=8)

    def test_mlp_gradients_and_diagnostics(self) -> None:
        """中心差分检查九个参数，并核对损失下降和诊断量。"""
        metrics, gradients = MODULE.mlp_evaluate(MODULE.INITIAL_PARAMETERS, True)
        self.assertIsNotNone(gradients)
        step = 1e-5
        for index, analytic in enumerate(gradients):
            plus = list(MODULE.INITIAL_PARAMETERS)
            minus = list(MODULE.INITIAL_PARAMETERS)
            plus[index] += step
            minus[index] -= step
            high, _ = MODULE.mlp_evaluate(tuple(plus))
            low, _ = MODULE.mlp_evaluate(tuple(minus))
            numerical = (high.loss - low.loss) / (2 * step)
            self.assertAlmostEqual(numerical, analytic, delta=1e-8)
        parameters = MODULE.INITIAL_PARAMETERS
        for _ in range(40):
            parameters, gradient_norm, update_norm = MODULE.mlp_train_step(parameters, 0.1)
        final, _ = MODULE.mlp_evaluate(parameters)
        self.assertLess(final.loss, metrics.loss)
        self.assertGreater(gradient_norm, 0)
        self.assertGreater(update_norm, 0)
        self.assertLessEqual(final.activation_minimum, final.activation_maximum)

    def test_overlarge_rate_is_detected_without_state_change(self) -> None:
        """检测有限但发散的轨迹，并拒绝会产生无穷数的更新。"""
        point = (1.0, 1.0)
        loss0, gradient = MODULE.quadratic_evaluate(point)
        point, _ = MODULE.optimizer_step(point, gradient, "sgd", 1.0)
        loss1, gradient = MODULE.quadratic_evaluate(point)
        self.assertGreater(loss1, loss0)
        previous_y = point[1]
        point, _ = MODULE.optimizer_step(point, gradient, "sgd", 1.0)
        loss2, gradient = MODULE.quadratic_evaluate(point)
        self.assertLess(previous_y * point[1], 0)
        self.assertGreater(loss2, loss1)
        previous_y = point[1]
        point, _ = MODULE.optimizer_step(point, gradient, "sgd", 1.0)
        loss3, gradient = MODULE.quadratic_evaluate(point)
        self.assertLess(previous_y * point[1], 0)
        self.assertGreater(loss3, loss2)
        saved_point = point
        state = MODULE.OptimizerState()
        with self.assertRaises(ValueError):
            MODULE.optimizer_step(
                point, gradient, "momentum", float.fromhex("0x1.fffffffffffffp+1023"), state)
        self.assertEqual(point, saved_point)
        self.assertEqual(state.velocity, [0.0, 0.0])
        with self.assertRaises(ValueError):
            MODULE.optimizer_step(point, gradient, "sgd", math.nan)
        with self.assertRaises(ValueError):
            MODULE.quadratic_evaluate((1.0,))
        with self.assertRaises(ValueError):
            MODULE.quadratic_evaluate((math.nan, 1.0))
        with self.assertRaises(ValueError):
            MODULE.mlp_evaluate(MODULE.INITIAL_PARAMETERS[:-1])
        invalid_parameters = list(MODULE.INITIAL_PARAMETERS)
        invalid_parameters[0] = math.inf
        with self.assertRaises(ValueError):
            MODULE.mlp_evaluate(tuple(invalid_parameters))

    def test_c_python_csv_and_decay_distinction(self) -> None:
        """比较两语言数值，并验证 L2 与解耦衰减进入不同更新路径。"""
        output = subprocess.check_output(
            [str(ROOT / "build/ml/38_optimization_diagnostics"), "--csv"],
            cwd=ROOT, text=True,
        )
        c_values = {row["field"]: float(row["value"])
                    for row in csv.DictReader(io.StringIO(output))}
        python_values = MODULE.results()
        self.assertEqual(c_values.keys(), python_values.keys())
        for key in c_values:
            self.assertAlmostEqual(c_values[key], python_values[key], delta=1e-10)

        coupled = (2.0, 0.0)
        decoupled = (2.0, 0.0)
        coupled_state = MODULE.OptimizerState()
        decoupled_state = MODULE.OptimizerState()
        rate = 0.01
        decay = 0.1
        for _ in range(2):
            coupled_gradient = (0.5 + decay * coupled[0], 0.0)
            old_decoupled_parameter = decoupled[0]
            coupled, coupled_state = MODULE.optimizer_step(
                coupled, coupled_gradient, "adam", rate, coupled_state)
            decoupled, decoupled_state = MODULE.optimizer_step(
                decoupled, (0.5, 0.0), "adam", rate, decoupled_state)
            decoupled = (decoupled[0] - rate * decay * old_decoupled_parameter,
                         decoupled[1])
        self.assertNotAlmostEqual(coupled[0], decoupled[0], delta=1e-7)


if __name__ == "__main__":
    unittest.main()
