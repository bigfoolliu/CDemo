"""检验标量自动求导的边界、手算值、数值梯度及 C/Python 一致性。"""

from __future__ import annotations

import csv
import importlib.util
import math
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/python/ml/32_scalar_autodiff.py"
spec = importlib.util.spec_from_file_location("scalar_autodiff", SOURCE)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


class ScalarAutodiffTest(unittest.TestCase):
    """针对共享节点及固定 XOR 实验执行可重复测试。"""

    def test_expression_and_repeated_backward(self) -> None:
        """检查同节点双输入、分支汇合及重复反向。"""
        graph = module.Graph()
        x, y = graph.leaf(2), graph.leaf(3)
        square = graph.mul(x, x)
        product = graph.mul(x, y)
        output = graph.add(graph.add(square, product), graph.unary("exp", x))
        for _ in range(2):
            graph.backward(output)
            self.assertAlmostEqual(graph.nodes[x].gradient, 7 + math.exp(2), places=12)
            self.assertEqual(graph.nodes[y].gradient, 2)
        step = 1e-5
        f = lambda value: value * value + value * 3 + math.exp(value)
        numeric = (f(2 + step) - f(2 - step)) / (2 * step)
        self.assertLess(abs(graph.nodes[x].gradient - numeric), 1e-8)
        log_node = graph.unary("log", graph.unary("exp", x))
        graph.backward(log_node)
        self.assertAlmostEqual(graph.nodes[x].gradient, 1.0)

    def test_failure_and_reset(self) -> None:
        """拒绝非法对数、非有限数、无效索引和满容量。"""
        graph = module.Graph(capacity=2)
        x = graph.leaf(-1)
        count = len(graph.nodes)
        with self.assertRaises(ValueError):
            graph.unary("log", x)
        self.assertEqual(len(graph.nodes), count)
        graph.leaf(2)
        with self.assertRaises(ValueError):
            graph.add(0, 1)
        with self.assertRaises(ValueError):
            graph.backward(2)
        graph.reset()
        self.assertEqual(graph.leaf(3), 0)
        with self.assertRaises(ValueError):
            graph.leaf(float("nan"))
        with self.assertRaises(ValueError):
            graph.mul(-1, 0)
        with self.assertRaises(ValueError):
            graph.unary("exp", graph.leaf(1000))
        with self.assertRaises(ValueError):
            graph.push("unknown", None, None, 1)

    def test_xor_gradients_and_training(self) -> None:
        """九个梯度对照手写结果与中心差分，并检查训练拟合。"""
        parameters = module.INITIAL
        step = 1e-5
        for sample in module.SAMPLES:
            automatic = module.xor_sample(parameters, *sample)
            manual = module.manual_sample(parameters, *sample)
            for actual, expected in zip(automatic[:2], manual[:2]):
                self.assertAlmostEqual(actual, expected, places=12)
            for index, (actual, expected) in enumerate(zip(automatic[2], manual[2])):
                self.assertAlmostEqual(actual, expected, places=12)
                high, low = list(parameters), list(parameters)
                high[index] += step
                low[index] -= step
                numeric = (module.xor_sample(tuple(high), *sample)[1] -
                           module.xor_sample(tuple(low), *sample)[1]) / (2 * step)
                self.assertLess(abs(actual - numeric), 1e-8 + 1e-6 * abs(numeric))
        initial = module.mean_loss(parameters)
        for _ in range(module.EPOCHS):
            parameters = module.train_epoch(parameters)
        self.assertLess(module.mean_loss(parameters), initial)
        self.assertLess(module.mean_loss(parameters), 0.003)
        for sample in module.SAMPLES:
            prediction = module.xor_sample(parameters, *sample)[0]
            self.assertLess(abs(prediction - sample[2]), 0.1)

    def test_c_python_csv(self) -> None:
        """在相同输入和初值下比较完整训练后的 C/Python 输出。"""
        c_output = subprocess.check_output(
            [str(ROOT / "build/ml/32_scalar_autodiff"), "--csv"], text=True, cwd=ROOT
        )
        python_output = subprocess.check_output(
            [sys.executable, str(SOURCE), "--csv"], text=True, cwd=ROOT
        )
        c_rows = dict(csv.reader(c_output.splitlines()[1:]))
        python_rows = dict(csv.reader(python_output.splitlines()[1:]))
        self.assertEqual(c_rows.keys(), python_rows.keys())
        for field in c_rows:
            self.assertAlmostEqual(float(c_rows[field]), float(python_rows[field]), places=12)


if __name__ == "__main__":
    unittest.main()
