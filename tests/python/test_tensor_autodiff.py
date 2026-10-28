"""验证固定张量图、手算梯度、边界和 C/Python 对照。"""

import csv
import importlib.util
import io
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src/python/ml/37_tensor_autodiff.py"
SPEC = importlib.util.spec_from_file_location("tensor_autodiff", SOURCE)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class TensorAutodiffTests(unittest.TestCase):
    """按 M02 固定输入检查 M08 自动求导的结果与契约。"""

    def test_dense_hand_values(self) -> None:
        """检查预测、平均半平方损失、广播偏置及一次更新。"""
        values = MODULE.dense_run()
        self.assertEqual(values["loss"], 1.84375)
        self.assertEqual([values[f"prediction_{i}"] for i in range(4)],
                         [5.5, -0.5, 1.5, 1.0])
        self.assertEqual([values[f"weight_grad_{i}"] for i in range(4)],
                         [0.5, -0.125, 2.125, -0.25])
        self.assertEqual([values[f"bias_grad_{i}"] for i in range(2)],
                         [1.25, -0.125])
        self.assertAlmostEqual(values["updated_weight_2"], 1.7875)
        self.assertAlmostEqual(values["updated_bias_0"], 0.375)

    def test_ten_central_differences(self) -> None:
        """逐个检查输入、权重及偏置的十项梯度。"""
        base = MODULE.dense_run()
        step = 1e-5
        for field, original, prefix in (("inputs", MODULE.INPUTS, "input_grad"),
                                        ("weights", MODULE.WEIGHTS, "weight_grad"),
                                        ("bias", MODULE.BIAS, "bias_grad")):
            for index in range(len(original)):
                plus = list(original)
                minus = list(original)
                plus[index] += step
                minus[index] -= step
                inputs = {field: tuple(plus)}
                high = MODULE.dense_run(**inputs)["loss"]
                inputs[field] = tuple(minus)
                low = MODULE.dense_run(**inputs)["loss"]
                self.assertAlmostEqual((high - low) / (2 * step),
                                       base[f"{prefix}_{index}"], delta=1e-8)

    def test_shared_broadcast_and_lifecycle(self) -> None:
        """检查共享节点累加、广播反向求和、清零与释放。"""
        graph = MODULE.Graph(capacity=8)
        matrix = graph.leaf(2, 2, (1., 2., 3., 4.))
        bias = graph.leaf(1, 2, (0.5, -0.5))
        added = graph.binary("add", matrix, bias)
        total = graph.unary("sum", added)
        graph.backward(total)
        self.assertEqual(graph.nodes[bias].gradients, [2, 2])
        graph.backward(total)
        self.assertEqual(graph.nodes[bias].gradients, [2, 2])
        with self.assertRaises(ValueError):
            graph.binary("matmul", matrix, bias)
        graph.release()
        self.assertEqual(len(graph.nodes), 0)
        with self.assertRaises(ValueError):
            graph.backward(total)

        shared = MODULE.Graph(capacity=4)
        leaf = shared.leaf(1, 2, (-1., 2.))
        doubled = shared.binary("add", leaf, leaf)
        relu = shared.unary("relu", doubled)
        summed = shared.unary("sum", relu)
        shared.backward(summed)
        self.assertEqual(shared.nodes[leaf].gradients, [0, 2])
        with self.assertRaises(ValueError):
            shared.leaf(1, 1, (1.,))

    def test_invalid_and_cross_language(self) -> None:
        """拒绝非有限数与错误形状，并核对 C 的全部固定字段。"""
        graph = MODULE.Graph(capacity=3)
        with self.assertRaises(ValueError):
            graph.leaf(2, 2, (1., 2.))
        with self.assertRaises(ValueError):
            graph.leaf(1, 1, (float("nan"),))
        self.assertEqual(len(graph.nodes), 0)
        a = graph.leaf(1, 2, (1., 2.))
        with self.assertRaises(ValueError):
            graph.binary("matmul", a, a)
        self.assertEqual(len(graph.nodes), 1)
        output = subprocess.check_output(
            [str(ROOT / "build/ml/37_tensor_autodiff"), "--csv"], cwd=ROOT, text=True)
        c_values = {row["field"]: float(row["value"])
                    for row in csv.DictReader(io.StringIO(output))}
        python_values = MODULE.dense_run()
        self.assertEqual(c_values.keys(), python_values.keys())
        for key in c_values:
            self.assertAlmostEqual(c_values[key], python_values[key], delta=1e-12)


if __name__ == "__main__":
    unittest.main()
