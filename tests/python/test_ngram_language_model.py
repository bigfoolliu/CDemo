"""验证一元/二元语言模型的计数、评分、采样和跨语言一致性。"""

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
SOURCE = ROOT / "src/python/ml/43_ngram_language_model.py"
SPEC = importlib.util.spec_from_file_location("ngram_language_model", SOURCE)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class NgramLanguageModelTests(unittest.TestCase):
    """核对特殊 token 约定、概率质量、留出指标和随机采样。"""

    def setUp(self) -> None:
        """为各测试构造固定训练集和留出集。"""
        self.training = MODULE.encode_dataset(MODULE.TRAINING_TEXT)
        self.testing = MODULE.encode_dataset(MODULE.TESTING_TEXT)
        self.unigram = MODULE.train_model(self.training, "unigram", len(MODULE.VOCABULARY))
        self.bigram = MODULE.train_model(self.training, "bigram", len(MODULE.VOCABULARY))

    def test_document_encoding_adds_boundaries_and_maps_oov(self) -> None:
        """BOS 与 EOS 边界明确，未知单词统一编码为 UNK。"""
        document = MODULE.encode_document("YOU rust .")
        self.assertEqual(document.tokens, (MODULE.BOS, 11, MODULE.UNK, 3, MODULE.EOS))
        self.assertEqual(MODULE.encode_document("").tokens, (MODULE.BOS, MODULE.EOS))

    def test_document_capacity_is_enforced(self) -> None:
        """超过固定 token 容量的文档必须明确失败。"""
        text = "x " * MODULE.MAX_SEQUENCE_LENGTH
        with self.assertRaises(ValueError):
            MODULE.encode_document(text)

    def test_counts_exclude_bos_as_target_and_count_eos(self) -> None:
        """BOS 不作为预测目标，每篇文档的 EOS 都计入一元目标。"""
        self.assertEqual(self.unigram.unigram_count[MODULE.BOS], 0)
        self.assertEqual(self.unigram.unigram_count[MODULE.EOS], len(self.training))
        self.assertEqual(self.unigram.token_count, 30)

    def test_every_bigram_row_sums_to_one_including_unseen_context(self) -> None:
        """加性平滑使所有上下文行归一，未见上下文得到均匀分布。"""
        for history in range(self.bigram.vocabulary_size):
            row = MODULE.next_probabilities(self.bigram, history)
            self.assertAlmostEqual(sum(row), 1.0, places=14)
            self.assertEqual(row[MODULE.BOS], 0.0)
            positive = all(
                row[token] > 0.0 for token in range(len(row)) if token != MODULE.BOS
            )
            self.assertTrue(positive)
        unseen = MODULE.next_probabilities(self.bigram, MODULE.TOKEN_TO_ID["study"])
        self.assertAlmostEqual(unseen[MODULE.EOS], 1 / (len(MODULE.VOCABULARY) - 1))

    def test_unseen_word_has_nonzero_probability(self) -> None:
        """训练集之外的 rust 映射为 UNK，留出 NLL 保持有限。"""
        self.assertIn(MODULE.UNK, self.testing[1].tokens)
        for model in (self.unigram, self.bigram):
            metrics = MODULE.evaluate(model, self.testing)
            self.assertEqual(metrics.token_count, 10)
            self.assertTrue(math.isfinite(metrics.nll))
            self.assertGreater(metrics.perplexity, 1.0)

    def test_documents_are_split_before_tokenization(self) -> None:
        """训练/测试分别来自两个原始文档数组，不从同文档切窗口。"""
        self.assertTrue(set(MODULE.TRAINING_TEXT).isdisjoint(MODULE.TESTING_TEXT))
        self.assertEqual(len(self.training), 6)
        self.assertEqual(len(self.testing), 2)

    def test_hand_calculation_matches_closed_form(self) -> None:
        """逐项核对加一后的四类概率和两个句子概率。"""
        values = MODULE.hand_example()
        unigram = values["unigram"]
        bigram = values["bigram"]
        self.assertEqual(unigram[:4], (2 / 7, 2 / 7, 2 / 7, 1 / 7))
        self.assertAlmostEqual(unigram[4], 8 / 343)
        self.assertAlmostEqual(unigram[5], -math.log(8 / 343) / 3)
        self.assertAlmostEqual(unigram[6], 3.5)
        self.assertEqual(bigram[:3], (2 / 5, 2 / 5, 2 / 5))
        self.assertAlmostEqual(bigram[3], 8 / 125)
        self.assertAlmostEqual(bigram[4], -math.log(8 / 125) / 3)
        self.assertAlmostEqual(bigram[5], 2.5)

    def test_sampling_is_reproducible_and_bounded(self) -> None:
        """固定种子采样可复现，未遇 EOS 时严格遵守长度上限。"""
        first = MODULE.sample(self.bigram, 12345, 5)
        second = MODULE.sample(self.bigram, 12345, 5)
        self.assertEqual(first, second)
        self.assertLessEqual(len(first), 5)
        self.assertNotIn(MODULE.BOS, first)
        with self.assertRaises(ValueError):
            MODULE.sample(self.bigram, 1, 0)

    def test_invalid_models_and_documents_fail_explicitly(self) -> None:
        """非法类型、边界符和不匹配的词表不会静默进入训练。"""
        with self.assertRaises(ValueError):
            MODULE.train_model(self.training, "trigram", len(MODULE.VOCABULARY))
        with self.assertRaises(ValueError):
            MODULE.train_model((MODULE.Document((MODULE.BOS, 3)),), "bigram", 5)
        with self.assertRaises(ValueError):
            MODULE.evaluate(self.bigram, ())

    def test_c_and_python_outputs_match_hand_and_test_metrics(self) -> None:
        """比较留出指标和手算夹具的 C/Python 中间数值。"""
        executable = Path(
            os.environ.get("NGRAM_LANGUAGE_MODEL_C", ROOT / "build/ml/43_ngram_language_model")
        )
        process = subprocess.run(
            [str(executable), "--csv"], check=True, capture_output=True, text=True, cwd=ROOT
        )
        c_rows = list(csv.reader(io.StringIO(process.stdout)))
        python_rows = list(csv.reader(io.StringIO(MODULE.csv_text())))
        self.assertEqual(len(c_rows), len(python_rows))
        for c_row, python_row in zip(c_rows, python_rows):
            self.assertEqual(c_row[:2], python_row[:2])
            if c_row[0] == "test":
                self.assertEqual(c_row[2], python_row[2])
                numeric_start = 3
            else:
                numeric_start = 2
            self.assertEqual(len(c_row), len(python_row))
            for left, right in zip(c_row[numeric_start:], python_row[numeric_start:]):
                self.assertLessEqual(abs(float(left) - float(right)), 2e-11)


if __name__ == "__main__":
    unittest.main()
