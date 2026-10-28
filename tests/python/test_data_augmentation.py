"""验证 D02 的图像变换、Softmax 梯度、数据隔离和跨语言结果。"""

import csv
import importlib.util
import io
import math
import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src/python/ml/42_data_augmentation.py"
SPEC = importlib.util.spec_from_file_location("data_augmentation", SOURCE)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class DataAugmentationTests(unittest.TestCase):
    """检查增强函数和固定小型消融实验。"""

    @classmethod
    def setUpClass(cls) -> None:
        """只训练一次 Python 对照结果，供测试复用。"""
        cls.results = MODULE.compute_results()

    def test_translation_preserves_label_and_source(self) -> None:
        """验证平移改变像素位置但保留标签和来源标识。"""
        pixels = [0.0] * MODULE.PIXELS
        pixels[3 * MODULE.SIDE + 3] = 1.0
        image = MODULE.Image(tuple(pixels), 2, 7)
        moved = MODULE.translate(image.pixels, 1, 0)
        self.assertIsNotNone(moved)
        self.assertEqual(moved[3 * MODULE.SIDE + 4], 1.0)
        transformed = MODULE.augment(image, "none", 9)
        self.assertEqual(transformed.target, image.target)
        self.assertEqual(transformed.source_id, image.source_id)
        self.assertIsNone(MODULE.translate(image.pixels, 2, 0))

    def test_noise_is_deterministic_and_clipped(self) -> None:
        """验证噪声固定种子可复现且像素裁切到合法区间。"""
        state, random_value = MODULE.random_unit(1)
        del state
        pixels, _ = MODULE.add_noise((0.5,) * MODULE.PIXELS, 1, 0.12)
        expected = 0.5 + 0.12 * (2 * random_value - 1)
        self.assertAlmostEqual(pixels[0], expected, places=15)
        self.assertEqual(pixels, MODULE.add_noise((0.5,) * MODULE.PIXELS, 1, 0.12)[0])
        lower, _ = MODULE.add_noise((0.0,) * MODULE.PIXELS, 1, 0.12)
        upper, _ = MODULE.add_noise((1.0,) * MODULE.PIXELS, 1, 0.12)
        self.assertTrue(all(0 <= value <= 1 for value in lower + upper))

    def test_bad_rotation_swaps_line_classes_but_keeps_the_label(self) -> None:
        """展示错误旋转会制造图像方向与标签不一致。"""
        horizontal = next(image for image in MODULE.make_dataset(3, 2025, 1) if image.target == 2)
        rotated = MODULE.augment(horizontal, "bad_rotate", 1)
        expected = tuple(
            horizontal.pixels[(MODULE.SIDE - 1 - column) * MODULE.SIDE + row]
            for row in range(MODULE.SIDE)
            for column in range(MODULE.SIDE)
        )
        self.assertEqual(rotated.pixels, expected)
        self.assertEqual(rotated.target, horizontal.target)
        self.assertNotEqual(rotated.pixels, horizontal.pixels)

    def test_invalid_inputs_fail_without_mutating_source(self) -> None:
        """拒绝非法尺寸、模式、像素和噪声幅度。"""
        image = MODULE.make_dataset(3, 2025, 1)[0]
        with self.assertRaises(ValueError):
            MODULE.make_dataset(4, 2025, 1)
        with self.assertRaises(ValueError):
            MODULE.augment(image, "unknown", 1)
        with self.assertRaises(ValueError):
            MODULE.add_noise(image.pixels, 1, math.inf)
        bad_image = MODULE.Image((math.nan,) + image.pixels[1:], image.target, image.source_id)
        with self.assertRaises(ValueError):
            MODULE.augment(bad_image, "none", 1)
        self.assertEqual(image, MODULE.make_dataset(3, 2025, 1)[0])

    def test_dataset_splits_are_independent_and_balanced(self) -> None:
        """确认先生成彼此分组隔离的数据，再从训练样本产生视图。"""
        training = MODULE.make_dataset(72, 2025, 1)
        validation = MODULE.make_dataset(36, 2026, 2)
        testing = MODULE.make_dataset(36, 2027, 3)
        groups = [{image.source_id // 1024 for image in split}
                  for split in (training, validation, testing)]
        self.assertEqual(groups, [{1}, {2}, {3}])
        for split in (training, validation, testing):
            counts = [sum(image.target == category for image in split)
                      for category in range(MODULE.CLASSES)]
            self.assertEqual(counts, [len(split) // MODULE.CLASSES] * MODULE.CLASSES)
        original = training[0]
        transformed = MODULE.augment(original, "combined", 42)
        self.assertEqual(original.source_id, transformed.source_id)
        self.assertEqual(original, training[0])

    def test_softmax_gradient_matches_central_difference(self) -> None:
        """对 192 个像素权重和 3 个偏置逐项执行中心差分。"""
        image = MODULE.Image(tuple(1.0 if i == 27 else 0.0 for i in range(MODULE.PIXELS)), 0, 1)
        weights, biases = MODULE.initialize(17)
        analytic, _ = MODULE.gradient(weights, biases, image)
        step = 1e-6
        for parameter in range(MODULE.PARAMETERS):
            plus_weights = weights.copy()
            minus_weights = weights.copy()
            plus_biases = biases.copy()
            minus_biases = biases.copy()
            if parameter < MODULE.WEIGHTS:
                plus_weights[parameter] += step
                minus_weights[parameter] -= step
            else:
                bias = parameter - MODULE.WEIGHTS
                plus_biases[bias] += step
                minus_biases[bias] -= step
            high = MODULE.forward(plus_weights, plus_biases, image)[1]
            low = MODULE.forward(minus_weights, minus_biases, image)[1]
            numerical = (high - low) / (2 * step)
            self.assertLess(abs(numerical - analytic[parameter]), 2e-7)

    def test_hand_example_matches_manual_softmax_update(self) -> None:
        """核对手算结果并确认 C/Python 使用相同输入参数。"""
        lines = MODULE.hand_example().splitlines()
        self.assertIn("0.943108 0.060217", lines[1])
        self.assertIn("probabilities=[0.401759578533, 0.328932922289, 0.269307499178]",
                      "\n".join(lines))
        self.assertIn("loss=0.911901432624", "\n".join(lines))
        self.assertIn(
            "one SGD step (rate=0.08)=[0.247859233717, -0.026314633783, -0.221544599934]",
            "\n".join(lines),
        )
        executable = Path(os.environ.get("AUGMENTATION_C", ROOT / "build/ml/42_data_augmentation"))
        process = subprocess.run(
            [str(executable), "--hand"], check=True, capture_output=True, text=True, cwd=ROOT
        )
        self.assertIn("logits=[0.200000000000, 0.000000000000, -0.200000000000]",
                      process.stdout)
        self.assertIn("\n".join(lines[1:]), process.stdout)

    def test_runs_share_the_same_budget_and_reproduce(self) -> None:
        """核对所有增强条件的更新预算和同种子重现性。"""
        self.assertEqual(MODULE.EPOCHS * MODULE.TRAIN_COUNT // MODULE.BATCH_SIZE, 960)
        training = MODULE.make_dataset(72, 2025, 1)
        validation = MODULE.make_dataset(36, 2026, 2)
        testing = MODULE.make_dataset(36, 2027, 3)
        first = MODULE.run_one(training, validation, testing, "translation", 42)
        second = MODULE.run_one(training, validation, testing, "translation", 42)
        self.assertEqual(first, second)

    def test_c_and_python_ablation_rows_match(self) -> None:
        """逐条件、逐种子比较 C 与 Python 的验证和测试指标。"""
        executable = Path(os.environ.get("AUGMENTATION_C", ROOT / "build/ml/42_data_augmentation"))
        process = subprocess.run(
            [str(executable), "--csv"], check=True, capture_output=True, text=True, cwd=ROOT
        )
        c_rows = list(csv.DictReader(io.StringIO(process.stdout)))
        python_rows = MODULE.csv_rows(self.results)
        self.assertEqual(len(c_rows), len(python_rows))
        self.assertEqual([row["augmentation"] for row in c_rows],
                         [row["augmentation"] for row in python_rows])
        for c_row, python_row in zip(c_rows, python_rows):
            self.assertEqual(c_row["seed"], python_row["seed"])
            for field in python_row:
                if field in ("augmentation", "seed"):
                    continue
                self.assertLessEqual(abs(float(c_row[field]) - float(python_row[field])), 2e-9)


if __name__ == "__main__":
    unittest.main()
