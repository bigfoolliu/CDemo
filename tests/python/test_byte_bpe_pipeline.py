"""验证 UTF-8 字节 BPE、文档窗口边界与 C/Python 共享模型格式。"""

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src/python/ml/44_byte_bpe_pipeline.py"
SPEC = importlib.util.spec_from_file_location("byte_bpe_pipeline", SOURCE)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class ByteBpePipelineTests(unittest.TestCase):
    """检查字节往返、合并顺序、数据隔离、批次填充及版本文件。"""

    def setUp(self) -> None:
        """为每项测试建立仅从固定训练文档拟合的模型。"""
        self.model = MODULE.train_bpe(MODULE.TRAINING_DOCUMENTS)

    def test_utf8_byte_roundtrip_and_byte_length(self) -> None:
        """中文按 UTF-8 字节编码，解码恢复原字节且不误当字符数。"""
        text = "中文学习"
        raw = text.encode("utf-8")
        encoded = MODULE.encode_text(self.model, text)
        self.assertEqual(len(raw), 12)
        self.assertEqual(len(text), 4)
        self.assertEqual(MODULE.decode_bytes(self.model, encoded), raw)

    def test_empty_text_and_control_token_policy(self) -> None:
        """空文本得到 BOS/EOS，正文编码为空且控制符不能解码成字节。"""
        body = MODULE.encode_bytes(self.model, b"")
        document = MODULE.encode_document(self.model, MODULE.RawDocument(9, ""), "train")
        self.assertEqual(body, ())
        self.assertEqual(document.tokens, (MODULE.BOS_ID, MODULE.EOS_ID))
        self.assertEqual(MODULE.decode_bytes(self.model, body), b"")
        with self.assertRaises(ValueError):
            MODULE.decode_bytes(self.model, (MODULE.BOS_ID,))
        with self.assertRaises(ValueError):
            MODULE.train_bpe(())

    def test_arbitrary_bytes_roundtrip_including_zero_and_invalid_utf8(self) -> None:
        """底层字节接口不把 NUL 当终止符，也不要求结果为合法 UTF-8。"""
        raw = bytes((0, 255, 128))
        self.assertEqual(
            MODULE.decode_bytes(self.model, MODULE.encode_bytes(self.model, raw)), raw
        )

    def test_capacity_failures_are_explicit(self) -> None:
        """超过固定字节容量和未知 token 都明确失败。"""
        maximum = bytes(range(MODULE.MAX_DOC_BYTES))
        self.assertEqual(
            MODULE.decode_bytes(self.model, MODULE.encode_bytes(self.model, maximum)), maximum
        )
        with self.assertRaises(ValueError):
            MODULE.encode_bytes(self.model, bytes(MODULE.MAX_DOC_BYTES + 1))
        with self.assertRaises(ValueError):
            MODULE.decode_bytes(self.model, (MODULE.MERGE_ID_START + len(self.model.merges),))
        with self.assertRaises(ValueError):
            MODULE.train_bpe(
                MODULE.TRAINING_DOCUMENTS, max_merges=MODULE.MAX_MERGES + 1
            )

    def test_two_round_hand_example(self) -> None:
        """手算样本先合并 ab，再把相邻 ab 合并为 abab。"""
        example = MODULE.train_bpe(
            (MODULE.RawDocument(1, "ababab"),), max_merges=2, min_frequency=1
        )
        self.assertEqual(example.merges, (MODULE.Merge(97, 98, 3), MODULE.Merge(259, 259, 2)))
        self.assertEqual(MODULE.encode_text(example, "ababab"), (260, 259))

    def test_pair_count_tie_uses_token_id_lexicographic_order(self) -> None:
        """aba 中 ab 和 ba 同频时选择字典序较小的 token 对。"""
        tied = MODULE.train_bpe(
            (MODULE.RawDocument(1, "aba"),), max_merges=1, min_frequency=1
        )
        self.assertEqual(tied.merges, (MODULE.Merge(97, 98, 1),))

    def test_only_training_documents_fit_merge_frequencies(self) -> None:
        """验证文档加入拟合集会改变频次，生产模型却只使用训练文档。"""
        self.assertEqual(self.model.merges[0].frequency, 6)
        all_documents = (
            *MODULE.TRAINING_DOCUMENTS,
            *MODULE.VALIDATION_DOCUMENTS,
            *MODULE.TEST_DOCUMENTS,
        )
        leaked = MODULE.train_bpe(all_documents, max_merges=1, min_frequency=1)
        self.assertEqual(leaked.merges[0].frequency, 10)
        self.assertEqual(self.model.merges[0].frequency, 6)

    def test_windows_shift_targets_and_overlap_within_document(self) -> None:
        """相邻窗口重叠上下文，但每个目标都来自同一原文档的下一位置。"""
        byte_model = MODULE.empty_model()
        document = MODULE.encode_document(
            byte_model, MODULE.RawDocument(999, "abcdef"), "train"
        )
        windows = MODULE.make_windows(document)
        self.assertEqual(windows[0].inputs, (MODULE.BOS_ID, ord("a"), ord("b"), ord("c")))
        self.assertEqual(windows[0].targets, (ord("a"), ord("b"), ord("c"), ord("d")))
        self.assertEqual(windows[1].inputs[:2], windows[0].inputs[2:])
        self.assertEqual(windows[1].document_id, 999)
        self.assertEqual(windows[1].split, "train")

    def test_batches_pad_inputs_targets_and_mask(self) -> None:
        """短尾窗口用 PAD 填满上下文，并通过 mask 排除填充值损失。"""
        byte_model = MODULE.empty_model()
        document = MODULE.encode_document(
            byte_model, MODULE.RawDocument(999, "abcdef"), "train"
        )
        batches = MODULE.make_batches(MODULE.make_windows(document))
        last_batch = batches[-1]
        self.assertEqual(last_batch.inputs[-1], (ord("f"),) + (MODULE.PAD_ID,) * 3)
        self.assertEqual(last_batch.targets[-1], (MODULE.EOS_ID,) + (MODULE.PAD_ID,) * 3)
        self.assertEqual(last_batch.mask[-1], (1, 0, 0, 0))
        self.assertEqual(len(last_batch.document_ids), 2)

    def test_batches_reject_mixed_splits(self) -> None:
        """同一批次不能混入两个数据划分的窗口。"""
        window = MODULE.Window("train", 1, 0, (MODULE.BOS_ID,), (MODULE.EOS_ID,))
        other = MODULE.Window("test", 2, 0, (MODULE.BOS_ID,), (MODULE.EOS_ID,))
        with self.assertRaises(ValueError):
            MODULE.make_batches((window, other))

    def test_pipeline_preserves_unique_source_document_splits(self) -> None:
        """原始文档先分组，所有重叠窗口和批次保留唯一来源划分。"""
        pipeline = MODULE.build_pipeline(self.model)
        document_sets = [
            {item.source.document_id for item in pipeline[split][0]}
            for split, _ in MODULE.DATA_SPLITS
        ]
        self.assertEqual(len(document_sets[0] & document_sets[1]), 0)
        self.assertEqual(len(document_sets[0] & document_sets[2]), 0)
        self.assertEqual(len(document_sets[1] & document_sets[2]), 0)
        for split, _ in MODULE.DATA_SPLITS:
            documents, windows, batches = pipeline[split]
            allowed = {item.source.document_id for item in documents}
            self.assertTrue(all(window.split == split and window.document_id in allowed for window in windows))
            self.assertTrue(all(batch.split == split for batch in batches))

    def test_versioned_serialization_roundtrip_and_corruption(self) -> None:
        """v1 模型可重建相同 token；未知版本、截断及尾随数据失败。"""
        encoded_file = MODULE.serialize_model(self.model)
        loaded = MODULE.deserialize_model(encoded_file)
        self.assertEqual(loaded.merges, self.model.merges)
        self.assertEqual(MODULE.encode_text(loaded, "你好"), MODULE.encode_text(self.model, "你好"))
        with self.assertRaises(ValueError):
            MODULE.deserialize_model(encoded_file[:5])
        bad_version = bytearray(encoded_file)
        bad_version[4:6] = (99).to_bytes(2, "little")
        with self.assertRaises(ValueError):
            MODULE.deserialize_model(bytes(bad_version))
        with self.assertRaises(ValueError):
            MODULE.deserialize_model(encoded_file + b"x")

    def test_c_and_python_csv_intermediates_match(self) -> None:
        """C 与 Python 的合并、UTF-8 文档、窗口和 PAD 批次逐行相同。"""
        executable = Path(
            os.environ.get("BYTE_BPE_C", ROOT / "build/ml/44_byte_bpe_pipeline")
        )
        completed = subprocess.run(
            [str(executable), "--csv"], check=True, capture_output=True, text=True, cwd=ROOT
        )
        self.assertEqual(completed.stdout, MODULE.csv_text())

    def test_c_python_versioned_files_are_shared_and_exclusive(self) -> None:
        """两种实现共享二进制格式，保存拒绝覆盖，加载结果与训练结果一致。"""
        executable = Path(
            os.environ.get("BYTE_BPE_C", ROOT / "build/ml/44_byte_bpe_pipeline")
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            c_model = Path(temporary_directory) / "c-model.bin"
            subprocess.run(
                [str(executable), "--save", str(c_model)],
                check=True, capture_output=True, text=True, cwd=ROOT,
            )
            original = c_model.read_bytes()
            self.assertEqual(original, MODULE.serialize_model(self.model))
            loaded_c = subprocess.run(
                [str(executable), "--load", str(c_model), "--csv"],
                check=True, capture_output=True, text=True, cwd=ROOT,
            )
            self.assertEqual(loaded_c.stdout, MODULE.csv_text())
            refused = subprocess.run(
                [str(executable), "--save", str(c_model)], capture_output=True, text=True, cwd=ROOT
            )
            self.assertNotEqual(refused.returncode, 0)
            self.assertEqual(c_model.read_bytes(), original)

            python_model = Path(temporary_directory) / "python-model.bin"
            subprocess.run(
                [sys.executable, str(SOURCE), "--save", str(python_model)],
                check=True, capture_output=True, text=True, cwd=ROOT,
            )
            self.assertEqual(python_model.read_bytes(), original)
            loaded_python = subprocess.run(
                [str(executable), "--load", str(python_model), "--csv"],
                check=True, capture_output=True, text=True, cwd=ROOT,
            )
            self.assertEqual(loaded_python.stdout, MODULE.csv_text())


if __name__ == "__main__":
    unittest.main()
