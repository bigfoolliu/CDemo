"""验证模型格式、故障安全、C/Python 共用夹具和独立进程推理。"""

import importlib.util
import math
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile
import unittest
import zlib

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src/python/ml/40_model_persistence.py"
FIXTURE = ROOT / "tests/fixtures/ml/model_v1.bin"
SPEC = importlib.util.spec_from_file_location("model_persistence", SOURCE)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
C_PROGRAM = Path(os.environ.get(
    "MODEL_PERSISTENCE_C", ROOT / "build/ml/40_model_persistence"))


def with_checksum(data: bytearray) -> bytes:
    """为损坏测试重新计算 CRC，让测试进入目标字段校验分支。

    :param data: 固定长度的可变模型字节。
    :return: 更新校验和后的不可变模型字节。
    """
    struct.pack_into("<I", data, len(data) - 4, zlib.crc32(data[:-4]))
    return bytes(data)


class ModelPersistenceTests(unittest.TestCase):
    """验证推理状态持久化前后的等价性及失败安全。"""

    def test_binary_loss_gradient_matches_central_difference(self) -> None:
        """用中心差分校验 BCE 对 logit 的解析导数和失败状态。"""
        logit = 0.4
        step = 1e-6
        for label in (0, 1):
            loss, gradient = MODULE.binary_loss_gradient(logit, label)
            loss_before, _ = MODULE.binary_loss_gradient(logit - step, label)
            loss_after, _ = MODULE.binary_loss_gradient(logit + step, label)
            numeric = (loss_after - loss_before) / (2.0 * step)
            self.assertGreaterEqual(loss, 0.0)
            self.assertAlmostEqual(numeric, gradient, delta=1e-8)
        with self.assertRaises(ValueError):
            MODULE.binary_loss_gradient(math.nan, 1)
        with self.assertRaises(ValueError):
            MODULE.binary_loss_gradient(0.0, 2)

    def test_common_fixture_matches_hand_calculation_in_both_languages(self) -> None:
        """两种语言从同一二进制夹具得到相同标准化值和概率。"""
        model = MODULE.load_model(FIXTURE)
        features = (4.0, 14.0)
        python_result = MODULE.predict(model, features)
        self.assertEqual(python_result[0], (1.0, 1.0))
        self.assertAlmostEqual(python_result[1], 0.15, places=14)
        self.assertAlmostEqual(python_result[2], 0.5374298453437496, places=14)
        self.assertEqual(python_result[3], 1)

        result = subprocess.run(
            [str(C_PROGRAM), "predict", str(FIXTURE), "4", "14"],
            cwd=ROOT, capture_output=True, text=True, check=True)
        c_values = re.search(r"probability=([^ ]+) class=(\d+)", result.stdout)
        self.assertIsNotNone(c_values)
        self.assertAlmostEqual(float(c_values.group(1)), python_result[2], places=14)
        self.assertEqual(int(c_values.group(2)), python_result[3])

    def test_bad_files_are_rejected_without_replacing_valid_state(self) -> None:
        """截断、错误版本/维度、非有限参数和 CRC 错误均被拒绝。"""
        valid = bytearray(FIXTURE.read_bytes())
        wrong_version = bytearray(valid)
        struct.pack_into("<I", wrong_version, 8, MODULE.FORMAT_VERSION + 1)
        wrong_dimension = bytearray(valid)
        struct.pack_into("<I", wrong_dimension, 12, MODULE.FEATURE_COUNT + 1)
        non_finite = bytearray(valid)
        struct.pack_into("<d", non_finite, MODULE.HEADER.size, math.inf)
        cases = {
            "truncated": bytes(valid[:-1]),
            "version": with_checksum(wrong_version),
            "dimension": with_checksum(wrong_dimension),
            "non_finite": with_checksum(non_finite),
            "checksum": bytes(valid[:-1] + bytes([valid[-1] ^ 1])),
        }
        original = MODULE.load_model(FIXTURE)
        original_prediction = MODULE.predict(original, (4.0, 14.0))

        with tempfile.TemporaryDirectory() as directory:
            for name, contents in cases.items():
                with self.subTest(case=name):
                    path = Path(directory) / f"{name}.bin"
                    path.write_bytes(contents)
                    with self.assertRaises(ValueError):
                        MODULE.load_model(path)
                    process = subprocess.run(
                        [str(C_PROGRAM), "predict", str(path), "4", "14"],
                        cwd=ROOT, capture_output=True, text=True)
                    self.assertNotEqual(process.returncode, 0)

        self.assertEqual(MODULE.load_model(FIXTURE), original)
        self.assertEqual(MODULE.predict(original, (4.0, 14.0)), original_prediction)

    def test_training_save_then_two_new_processes_infer(self) -> None:
        """训练进程保存后，独立 C/Python 进程加载并复现预测。"""
        features = (5.2, 64.0)
        trained = MODULE.train_from_csv()
        expected = MODULE.predict(trained, features)
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "trained.bin"
            train = subprocess.run(
                [str(C_PROGRAM), "train", str(model_path)],
                cwd=ROOT, capture_output=True, text=True, check=True)
            self.assertIn("已保存版本 1", train.stdout)

            c_inference = subprocess.run(
                [str(C_PROGRAM), "predict", str(model_path), "5.2", "64"],
                cwd=ROOT, capture_output=True, text=True, check=True)
            c_match = re.search(r"standardized=([^, ]+),([^ ]+) logit=([^ ]+) "
                                r"probability=([^ ]+) class=(\d+)", c_inference.stdout)
            self.assertIsNotNone(c_match)
            self.assertAlmostEqual(float(c_match.group(1)), expected[0][0], places=12)
            self.assertAlmostEqual(float(c_match.group(2)), expected[0][1], places=12)
            self.assertAlmostEqual(float(c_match.group(3)), expected[1], places=12)
            self.assertAlmostEqual(float(c_match.group(4)), expected[2], places=12)
            self.assertEqual(int(c_match.group(5)), expected[3])

            python_inference = subprocess.run(
                [sys.executable, str(SOURCE), "predict", str(model_path), "5.2", "64"],
                cwd=ROOT, capture_output=True, text=True, check=True)
            python_match = re.search(r"probability=([^ ]+) class=(\d+)",
                                     python_inference.stdout)
            self.assertIsNotNone(python_match)
            self.assertAlmostEqual(float(python_match.group(1)), expected[2], places=12)
            self.assertEqual(int(python_match.group(2)), expected[3])
            loaded_result = MODULE.predict(MODULE.load_model(model_path), features)
            for left, right in zip((*loaded_result[0], *loaded_result[1:3]),
                                   (*expected[0], *expected[1:3])):
                self.assertAlmostEqual(left, right, places=12)
            self.assertEqual(loaded_result[3], expected[3])

    def test_existing_model_is_never_overwritten(self) -> None:
        """C 与 Python 的训练保存入口均使用独占创建并保留原字节。"""
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "keep.bin"
            original = b"user model must remain untouched"
            model_path.write_bytes(original)
            c_train = subprocess.run(
                [str(C_PROGRAM), "train", str(model_path)],
                cwd=ROOT, capture_output=True, text=True)
            self.assertNotEqual(c_train.returncode, 0)
            self.assertEqual(model_path.read_bytes(), original)
            with self.assertRaises(FileExistsError):
                MODULE.save_model_exclusive(model_path, MODULE.train_from_csv())
            self.assertEqual(model_path.read_bytes(), original)

    def test_training_parameters_match_between_c_and_python(self) -> None:
        """共同 CSV 和固定轮数下，两种语言得到一致的预处理与权重。"""
        model = MODULE.train_from_csv()
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "trained.bin"
            subprocess.run([str(C_PROGRAM), "train", str(model_path)], cwd=ROOT,
                           capture_output=True, text=True, check=True)
            c_model = MODULE.load_model(model_path)
        for left, right in zip((*model.mean, *model.scale, *model.weights, model.bias),
                               (*c_model.mean, *c_model.scale, *c_model.weights,
                                c_model.bias)):
            self.assertAlmostEqual(left, right, places=11)


if __name__ == "__main__":
    unittest.main()
