"""检查批量全连接层的手算、数值梯度及 C/Python 对照。"""

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
PYTHON_SOURCE = ROOT / "src/python/ml/31_batch_dense_gradients.py"
C_PROGRAM = ROOT / "build/ml/31_batch_dense_gradients"
spec = importlib.util.spec_from_file_location("batch_dense_gradients", PYTHON_SOURCE)
if spec is None or spec.loader is None:
    raise RuntimeError("无法加载 Python 批量求导示例")
dense = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dense)


class BatchDenseGradientTests(unittest.TestCase):
    """对同一固定小批量执行数值与行为验证。"""

    def assert_gradient(self, analytic: float, numerical: float) -> None:
        """按照文档容差核对一个中心差分值。

        :param analytic: 解析梯度。
        :param numerical: 中心差分梯度。
        :return: 无。
        """
        self.assertTrue(math.isclose(analytic, numerical, rel_tol=1e-6, abs_tol=1e-8),
                        (analytic, numerical))

    def test_hand_calculation(self) -> None:
        """比较前向、反向与更新的手算结果。

        :return: 无。
        """
        forward = dense.dense_forward(dense.INITIAL_MODEL, dense.INITIAL_BATCH)
        gradients = dense.dense_backward(dense.INITIAL_MODEL, dense.INITIAL_BATCH, forward)
        self.assertEqual(forward.predictions, ((5.5, -0.5), (1.5, 1.0)))
        self.assertEqual(forward.loss, 1.84375)
        self.assertEqual(gradients.inputs, ((1.0, 1.6875), (0.375, 0.75)))
        self.assertEqual(gradients.weights, ((0.5, -0.125), (2.125, -0.25)))
        self.assertEqual(gradients.bias, (1.25, -0.125))
        updated = dense.dense_update(dense.INITIAL_MODEL, gradients, 0.1)
        self.assertEqual(updated.weights, ((0.95, -0.9875), (1.7875, 0.525)))
        self.assertEqual(updated.bias, (0.375, -0.4875))
        self.assertAlmostEqual(dense.dense_forward(updated, dense.INITIAL_BATCH).loss,
                               1.25767578125)

    def test_all_ten_gradients(self) -> None:
        """逐项检查输入、权重和偏置的中心差分。

        :return: 无。
        """
        model = dense.INITIAL_MODEL
        batch = dense.INITIAL_BATCH
        forward = dense.dense_forward(model, batch)
        gradients = dense.dense_backward(model, batch, forward)
        step = 1e-5
        for sample in range(2):
            for feature in range(2):
                modified = [list(row) for row in batch.inputs]
                modified[sample][feature] += step
                plus = dense.Batch(tuple(tuple(row) for row in modified), batch.targets)
                modified[sample][feature] -= 2.0 * step
                minus = dense.Batch(tuple(tuple(row) for row in modified), batch.targets)
                numerical = (dense.dense_forward(model, plus).loss -
                             dense.dense_forward(model, minus).loss) / (2.0 * step)
                self.assert_gradient(gradients.inputs[sample][feature], numerical)
        for feature in range(2):
            for output in range(2):
                modified = [list(row) for row in model.weights]
                modified[feature][output] += step
                plus = dense.Model(tuple(tuple(row) for row in modified), model.bias)
                modified[feature][output] -= 2.0 * step
                minus = dense.Model(tuple(tuple(row) for row in modified), model.bias)
                numerical = (dense.dense_forward(plus, batch).loss -
                             dense.dense_forward(minus, batch).loss) / (2.0 * step)
                self.assert_gradient(gradients.weights[feature][output], numerical)
        for output in range(2):
            modified = list(model.bias)
            modified[output] += step
            plus = dense.Model(model.weights, tuple(modified))
            modified[output] -= 2.0 * step
            minus = dense.Model(model.weights, tuple(modified))
            numerical = (dense.dense_forward(plus, batch).loss -
                         dense.dense_forward(minus, batch).loss) / (2.0 * step)
            self.assert_gradient(gradients.bias[output], numerical)

    def test_batch_mean_and_failures(self) -> None:
        """复制样本应保持平均参数梯度，并检查非法输入。

        :return: 无。
        """
        batch = dense.INITIAL_BATCH
        one = dense.Batch(batch.inputs[:1], batch.targets[:1])
        doubled = dense.Batch((batch.inputs[0], batch.inputs[0]),
                              (batch.targets[0], batch.targets[0]))
        first = dense.dense_forward(dense.INITIAL_MODEL, one)
        second = dense.dense_forward(dense.INITIAL_MODEL, doubled)
        first_gradients = dense.dense_backward(dense.INITIAL_MODEL, one, first)
        second_gradients = dense.dense_backward(dense.INITIAL_MODEL, doubled, second)
        self.assertEqual(first.loss, second.loss)
        self.assertEqual(first_gradients.weights, second_gradients.weights)
        self.assertEqual(first_gradients.bias, second_gradients.bias)
        self.assertEqual(tuple(2.0 * value for value in second_gradients.inputs[0]),
                         first_gradients.inputs[0])
        with self.assertRaises(ValueError):
            dense.dense_forward(dense.INITIAL_MODEL, dense.Batch((), ()))
        with self.assertRaises(ValueError):
            dense.dense_forward(dense.INITIAL_MODEL,
                                dense.Batch(((math.nan, 0.0),), ((0.0, 0.0),)))
        with self.assertRaises(ValueError):
            dense.dense_forward(dense.INITIAL_MODEL,
                                dense.Batch(((1.0,),), ((0.0, 0.0),)))
        with self.assertRaises(ValueError):
            dense.dense_forward(dense.INITIAL_MODEL,
                                dense.Batch(((1e308, 1e308),), ((0.0, 0.0),)))
        with self.assertRaises(ValueError):
            dense.dense_update(dense.INITIAL_MODEL, first_gradients, 0.0)
        self.assertEqual(dense.INITIAL_MODEL.weights, ((1.0, -1.0), (2.0, 0.5)))

    def test_c_python_csv(self) -> None:
        """对照两个程序的全部固定中间值。

        :return: 无。
        """
        c_result = subprocess.run([str(C_PROGRAM), "--csv"], check=True,
                                  capture_output=True, text=True, cwd=ROOT)
        py_result = subprocess.run([sys.executable, str(PYTHON_SOURCE), "--csv"],
                                   check=True, capture_output=True, text=True, cwd=ROOT)
        c_rows = list(csv.DictReader(io.StringIO(c_result.stdout)))
        py_rows = list(csv.DictReader(io.StringIO(py_result.stdout)))
        self.assertEqual(len(c_rows), 22)
        self.assertEqual(len(c_rows), len(py_rows))
        for actual, expected in zip(c_rows, py_rows):
            self.assertEqual(actual["field"], expected["field"])
            self.assertTrue(math.isclose(float(actual["value"]), float(expected["value"]),
                                          rel_tol=1e-12, abs_tol=1e-12))


if __name__ == "__main__":
    unittest.main()
