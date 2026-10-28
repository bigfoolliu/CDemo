#!/usr/bin/env python3
"""用标准库展示 UTF-8 字节往返、简化 BPE 与文档级数据管线。"""

from __future__ import annotations

import argparse
import csv
import io
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


BASE_VOCAB_SIZE = 256
BOS_ID = 256
EOS_ID = 257
PAD_ID = 258
MERGE_ID_START = 259
MAX_MERGES = 8
MAX_TRAIN_DOCUMENTS = 8
MAX_DOC_BYTES = 64
MAX_SEQUENCE_LENGTH = MAX_DOC_BYTES + 2
CONTEXT_LENGTH = 4
WINDOW_STRIDE = 2
BATCH_SIZE = 2
MIN_PAIR_FREQUENCY = 2
MODEL_MAGIC = b"BPE1"
MODEL_VERSION = 1


@dataclass(frozen=True)
class RawDocument:
    """保存原始文档 ID 与文本。"""

    document_id: int
    text: str


@dataclass(frozen=True)
class Merge:
    """保存一条按训练顺序应用的 BPE 合并规则。"""

    left: int
    right: int
    frequency: int


@dataclass(frozen=True)
class BPEModel:
    """保存基础字节、特殊符号和有序合并规则。"""

    token_bytes: tuple[bytes, ...]
    merges: tuple[Merge, ...]


@dataclass(frozen=True)
class EncodedDocument:
    """保存来源划分、原始字节与带边界符的 token 序列。"""

    split: str
    source: RawDocument
    raw_bytes: bytes
    tokens: tuple[int, ...]


@dataclass(frozen=True)
class Window:
    """保存单篇文档产生的下一 token 输入和目标。"""

    split: str
    document_id: int
    start: int
    inputs: tuple[int, ...]
    targets: tuple[int, ...]


@dataclass(frozen=True)
class Batch:
    """保存同一划分中定长填充后的窗口来源。"""

    split: str
    document_ids: tuple[int, ...]
    inputs: tuple[tuple[int, ...], ...]
    targets: tuple[tuple[int, ...], ...]
    mask: tuple[tuple[int, ...], ...]


TRAINING_DOCUMENTS = (
    RawDocument(101, "ababab"),
    RawDocument(102, "ababac"),
    RawDocument(103, "你好 ab"),
)
VALIDATION_DOCUMENTS = (
    RawDocument(201, "abab"),
    RawDocument(202, "中文"),
)
TEST_DOCUMENTS = (
    RawDocument(301, "ababa!"),
    RawDocument(302, "学AI"),
)
DATA_SPLITS = (
    ("train", TRAINING_DOCUMENTS),
    ("validation", VALIDATION_DOCUMENTS),
    ("test", TEST_DOCUMENTS),
)


def empty_model() -> BPEModel:
    """建立包含全部单字节和独立控制符的初始词表。"""
    tokens = tuple(bytes((value,)) for value in range(BASE_VOCAB_SIZE)) + (b"", b"", b"")
    return BPEModel(tokens, ())


def _apply_merge(sequence: Sequence[int], left: int, right: int, result: int) -> tuple[int, ...]:
    """从左到右非重叠地应用一条相邻 token 合并规则。

    :param sequence: 当前 token 序列。
    :param left: 待匹配的左 token ID。
    :param right: 待匹配的右 token ID。
    :param result: 合并后 token ID。
    :return: 应用一次合并后的不可变序列。
    """
    output: list[int] = []
    index = 0
    while index < len(sequence):
        if index + 1 < len(sequence) and sequence[index] == left and sequence[index + 1] == right:
            output.append(result)
            index += 2
        else:
            output.append(sequence[index])
            index += 1
    return tuple(output)


def train_bpe(
    documents: Sequence[RawDocument],
    max_merges: int = MAX_MERGES,
    min_frequency: int = MIN_PAIR_FREQUENCY,
) -> BPEModel:
    """只从传入的原始训练文档逐轮学习最高频相邻 token 对。

    :param documents: 唯一允许参与词表拟合的训练文档，不转移所有权。
    :param max_merges: 合并规则上限，范围为 0..MAX_MERGES。
    :param min_frequency: 一对 token 至少出现的次数，须大于零。
    :return: 含字节词表和按频率顺序排列的 BPE 模型。
    :raises ValueError: 输入为空、文档重复或参数超出固定容量时抛出。
    """
    if (
        not documents
        or len(documents) > MAX_TRAIN_DOCUMENTS
        or not 0 <= max_merges <= MAX_MERGES
        or min_frequency < 1
    ):
        raise ValueError("训练文档或 BPE 参数无效")
    ids = [document.document_id for document in documents]
    if len(set(ids)) != len(ids):
        raise ValueError("训练文档 ID 必须唯一")

    sequences: list[tuple[int, ...]] = []
    for document in documents:
        if not isinstance(document.text, str):
            raise TypeError("文档文本必须是字符串")
        raw = document.text.encode("utf-8")
        if len(raw) > MAX_DOC_BYTES:
            raise ValueError("文档 UTF-8 字节数超过固定容量")
        sequences.append(tuple(raw))

    token_bytes = list(empty_model().token_bytes)
    merges: list[Merge] = []
    for _ in range(max_merges):
        counts: dict[tuple[int, int], int] = {}
        for sequence in sequences:
            for pair in zip(sequence, sequence[1:]):
                counts[pair] = counts.get(pair, 0) + 1
        if not counts:
            break
        pair = min(counts, key=lambda item: (-counts[item], item[0], item[1]))
        frequency = counts[pair]
        if frequency < min_frequency:
            break

        result = MERGE_ID_START + len(merges)
        merged_bytes = token_bytes[pair[0]] + token_bytes[pair[1]]
        if len(merged_bytes) > MAX_DOC_BYTES:
            raise ValueError("合并后的 token 超过固定字节容量")
        merges.append(Merge(pair[0], pair[1], frequency))
        token_bytes.append(merged_bytes)
        sequences = [
            _apply_merge(sequence, pair[0], pair[1], result) for sequence in sequences
        ]
    return BPEModel(tuple(token_bytes), tuple(merges))


def encode_bytes(model: BPEModel, data: bytes) -> tuple[int, ...]:
    """先把每个字节编码为 0..255，再依次应用已学习的合并。

    :param model: 已训练或空的 BPE 模型。
    :param data: 原始字节串；允许空串，长度不得超过 MAX_DOC_BYTES。
    :return: 不含 BOS/EOS/PAD 的 token ID 序列。
    :raises TypeError: data 不是 bytes 时抛出。
    :raises ValueError: 输入超过固定容量或模型结构非法时抛出。
    """
    if not isinstance(data, bytes):
        raise TypeError("data 必须是 bytes")
    if len(data) > MAX_DOC_BYTES or len(model.merges) > MAX_MERGES:
        raise ValueError("输入或模型超过固定容量")
    sequence: tuple[int, ...] = tuple(data)
    for rank, merge in enumerate(model.merges):
        sequence = _apply_merge(
            sequence, merge.left, merge.right, MERGE_ID_START + rank
        )
    return sequence


def encode_text(model: BPEModel, text: str) -> tuple[int, ...]:
    """把 Unicode 文本编码为 UTF-8，再调用字节级 BPE 编码。

    :param model: 已训练或空的 BPE 模型。
    :param text: Unicode 字符串；不执行规范化或分词前处理。
    :return: 不含控制符的 token ID 序列。
    :raises TypeError: text 不是字符串时抛出。
    :raises ValueError: UTF-8 字节长度超过固定容量时抛出。
    """
    if not isinstance(text, str):
        raise TypeError("text 必须是字符串")
    return encode_bytes(model, text.encode("utf-8"))


def decode_bytes(model: BPEModel, tokens: Sequence[int]) -> bytes:
    """将 BPE token 展开为原始字节，拒绝混入控制符。

    :param model: 与编码时相同的 BPE 模型。
    :param tokens: 仅含字节 token 或已登记合并 token 的序列。
    :return: 解码后的原始字节，不进行 UTF-8 验证。
    :raises ValueError: token 越界、包含控制符或展开后超出容量时抛出。
    """
    output = bytearray()
    for token in tokens:
        if not isinstance(token, int) or not 0 <= token < len(model.token_bytes):
            raise ValueError("token ID 不在模型词表中")
        if token in (BOS_ID, EOS_ID, PAD_ID):
            raise ValueError("控制符不能作为文本字节解码")
        output.extend(model.token_bytes[token])
        if len(output) > MAX_DOC_BYTES:
            raise ValueError("解码结果超过固定容量")
    return bytes(output)


def encode_document(model: BPEModel, document: RawDocument, split: str) -> EncodedDocument:
    """独立编码一篇文档，并在子词序列外添加 BOS 和 EOS。

    :param model: 只使用训练集拟合的 BPE 模型。
    :param document: 有唯一来源 ID 的原始文档。
    :param split: train、validation 或 test。
    :return: 保留来源 ID、原始字节及带边界符的 token 序列。
    :raises ValueError: split 不合法、ID 无效或文档超过容量时抛出。
    """
    if split not in {"train", "validation", "test"} or document.document_id < 0:
        raise ValueError("文档划分或来源 ID 无效")
    raw = document.text.encode("utf-8")
    body = encode_bytes(model, raw)
    tokens = (BOS_ID, *body, EOS_ID)
    if len(tokens) > MAX_SEQUENCE_LENGTH:
        raise ValueError("编码后的文档超过固定序列容量")
    return EncodedDocument(split, document, raw, tokens)


def make_windows(document: EncodedDocument) -> tuple[Window, ...]:
    """只在当前原文档内部生成带重叠的 next-token 窗口。

    :param document: 已按原文档边界编码的单篇文档。
    :return: 步长小于上下文长度、可能覆盖相同 token 的窗口。
    """
    windows: list[Window] = []
    final_input_start = len(document.tokens) - 1
    for start in range(0, final_input_start, WINDOW_STRIDE):
        end = min(start + CONTEXT_LENGTH, final_input_start)
        inputs = document.tokens[start:end]
        targets = document.tokens[start + 1 : end + 1]
        windows.append(Window(document.split, document.source.document_id, start, inputs, targets))
    return tuple(windows)


def make_batches(windows: Sequence[Window]) -> tuple[Batch, ...]:
    """按单个 split 的窗口顺序分批，并记录 PAD 所需的有效长度信息。

    :param windows: 同一数据划分中已生成的窗口。
    :return: 每批最多 BATCH_SIZE 个窗口及其来源文档 ID。
    :raises ValueError: 一个批次试图混入多个数据划分时抛出。
    """
    batches: list[Batch] = []
    for start in range(0, len(windows), BATCH_SIZE):
        group = windows[start : start + BATCH_SIZE]
        if len({window.split for window in group}) > 1:
            raise ValueError("批次不能混合训练、验证和测试数据")
        input_rows = []
        target_rows = []
        mask_rows = []
        for window in group:
            padding = CONTEXT_LENGTH - len(window.inputs)
            input_rows.append(window.inputs + (PAD_ID,) * padding)
            target_rows.append(window.targets + (PAD_ID,) * padding)
            mask_rows.append((1,) * len(window.inputs) + (0,) * padding)
        batches.append(Batch(
            group[0].split,
            tuple(window.document_id for window in group),
            tuple(input_rows),
            tuple(target_rows),
            tuple(mask_rows),
        ))
    return tuple(batches)


def build_pipeline(
    model: BPEModel,
) -> dict[str, tuple[tuple[EncodedDocument, ...], tuple[Window, ...], tuple[Batch, ...]]]:
    """分别编码三个原始文档划分，再分别生成窗口和批次。

    :param model: 只由 TRAINING_DOCUMENTS 拟合的 BPE 模型。
    :return: 每个 split 的编码文档、窗口和批次，不混合文档来源。
    """
    pipeline = {}
    for split, documents in DATA_SPLITS:
        encoded = tuple(encode_document(model, item, split) for item in documents)
        windows = tuple(window for item in encoded for window in make_windows(item))
        pipeline[split] = (encoded, windows, make_batches(windows))
    return pipeline


def serialize_model(model: BPEModel) -> bytes:
    """生成小端序、带魔数和版本号的独占保存格式。

    :param model: 已训练模型。
    :return: BPE1 v1 格式的词表合并规则字节串。
    :raises ValueError: 规则数或频次超出文件格式范围时抛出。
    """
    if len(model.merges) > MAX_MERGES:
        raise ValueError("合并规则超过文件格式上限")
    output = bytearray(struct.pack("<4sHH", MODEL_MAGIC, MODEL_VERSION, len(model.merges)))
    for merge in model.merges:
        if not 0 < merge.frequency <= 0xFFFFFFFF:
            raise ValueError("合并频次超出文件格式范围")
        output.extend(struct.pack("<HHI", merge.left, merge.right, merge.frequency))
    return bytes(output)


def deserialize_model(data: bytes) -> BPEModel:
    """校验版本、规则引用和文件长度后重建合并词表。

    :param data: 完整的 BPE1 二进制模型文件。
    :return: 可编码与解码的模型。
    :raises ValueError: 文件截断、版本未知、引用非法或含尾随数据时抛出。
    """
    header_size = struct.calcsize("<4sHH")
    record_size = struct.calcsize("<HHI")
    if len(data) < header_size:
        raise ValueError("模型文件头不完整")
    magic, version, merge_count = struct.unpack_from("<4sHH", data)
    if magic != MODEL_MAGIC or version != MODEL_VERSION or merge_count > MAX_MERGES:
        raise ValueError("模型魔数、版本或规则数无效")
    if len(data) != header_size + merge_count * record_size:
        raise ValueError("模型文件长度不匹配")

    tokens = list(empty_model().token_bytes)
    merges: list[Merge] = []
    for rank in range(merge_count):
        offset = header_size + rank * record_size
        left, right, frequency = struct.unpack_from("<HHI", data, offset)
        if left >= len(tokens) or right >= len(tokens) or frequency == 0:
            raise ValueError("合并规则引用无效")
        if left in (BOS_ID, EOS_ID, PAD_ID) or right in (BOS_ID, EOS_ID, PAD_ID):
            raise ValueError("合并规则不能引用控制符")
        merged = tokens[left] + tokens[right]
        if not merged or len(merged) > MAX_DOC_BYTES:
            raise ValueError("合并 token 长度无效")
        merges.append(Merge(left, right, frequency))
        tokens.append(merged)
    return BPEModel(tuple(tokens), tuple(merges))


def _ids(values: Sequence[int]) -> str:
    """把 ID 序列格式化为跨语言对照用分号串。

    :param values: token 或 mask 数值序列。
    :return: 使用分号连接的十进制字符串。
    """
    return ";".join(str(value) for value in values)


def csv_text(model: BPEModel | None = None) -> str:
    """导出合并、文档、窗口和批次的可逐字段对照结果。

    :param model: 可选的已训练模型；为空时只从训练文档拟合。
    :return: 使用逗号分隔字段、换行分隔记录的文本。
    """
    fitted = model if model is not None else train_bpe(TRAINING_DOCUMENTS)
    pipeline = build_pipeline(fitted)
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    for rank, merge in enumerate(fitted.merges):
        writer.writerow((
            "merge", rank, merge.left, merge.right, merge.frequency, MERGE_ID_START + rank
        ))
    for split, _ in DATA_SPLITS:
        encoded, _, _ = pipeline[split]
        for item in encoded:
            body = item.tokens[1:-1]
            writer.writerow((
                "document", split, item.source.document_id, len(item.raw_bytes),
                _ids(body), item.raw_bytes.hex(), _ids(item.tokens),
            ))
    for split, _ in DATA_SPLITS:
        _, windows, _ = pipeline[split]
        for window in windows:
            mask = (1,) * len(window.inputs)
            writer.writerow((
                "window", split, window.document_id, window.start, len(window.inputs),
                _ids(window.inputs), _ids(window.targets), _ids(mask),
            ))
    for split, _ in DATA_SPLITS:
        _, _, batches = pipeline[split]
        for index, batch in enumerate(batches):
            input_rows = "|".join(_ids(row) for row in batch.inputs)
            target_rows = "|".join(_ids(row) for row in batch.targets)
            mask_rows = "|".join(_ids(row) for row in batch.mask)
            writer.writerow((
                "batch", split, index, len(batch.document_ids), _ids(batch.document_ids),
                input_rows, target_rows, mask_rows,
            ))
    return stream.getvalue()


def hand_example() -> str:
    """返回可复算的两轮 BPE 手算结果和并列规则。"""
    model = train_bpe((RawDocument(1, "ababab"),), max_merges=2, min_frequency=1)
    tie = train_bpe((RawDocument(2, "aba"),), max_merges=1, min_frequency=1)
    lines = [
        "文本 ababab：",
        "第 0 轮：ab 出现 3 次、ba 出现 2 次，先合并 (a,b)。",
    ]
    lines.append(
        "第 1 轮：新 token ab 的相邻对 (ab,ab) 出现 2 次，再合并为 abab。"
    )
    lines.append(f"计算得到规则：{model.merges[0]}；{model.merges[1]}。")
    lines.append("并列例子 aba：ab 与 ba 各出现 1 次；ID 字典序选择 (97,98)。")
    lines.append(f"计算得到并列规则：{tie.merges[0]}。")
    return "\n".join(lines)


def demo(model: BPEModel) -> str:
    """生成固定语料的简短运行摘要。

    :param model: 由训练文档拟合或从版本化文件加载的模型。
    :return: 显示 BPE 规则、UTF-8 往返与各划分窗口数的摘要。
    """
    pipeline = build_pipeline(model)
    lines = [
        "L02 UTF-8 字节与简化 BPE",
        f"训练得到 {len(model.merges)} 条合并规则：",
    ]
    for rank, merge in enumerate(model.merges):
        lines.append(
            f"  rank={rank} pair=({merge.left},{merge.right}) "
            f"frequency={merge.frequency} result={MERGE_ID_START + rank}"
        )
    chinese = "你好".encode("utf-8")
    encoded = encode_bytes(model, chinese)
    lines.append(
        f"中文字节：{chinese.hex()} -> token IDs {_ids(encoded)} "
        f"-> {decode_bytes(model, encoded).hex()}"
    )
    for split, _ in DATA_SPLITS:
        documents, windows, batches = pipeline[split]
        lines.append(
            f"{split}: documents={len(documents)} windows={len(windows)} batches={len(batches)}"
        )
    lines.append(
        "BOS/EOS/PAD 为独立控制符；字节词表覆盖全部 256 种字节，没有 UNK。"
    )
    return "\n".join(lines)


def main() -> int:
    """解析命令行选项并运行教学演示。"""
    parser = argparse.ArgumentParser(description=__doc__)
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--csv", action="store_true", help="输出 C/Python 逐字段对照记录")
    output.add_argument("--hand", action="store_true", help="显示两轮 BPE 手算过程")
    parser.add_argument("--save", type=Path, help="独占创建版本化模型文件")
    parser.add_argument("--load", type=Path, help="加载版本化模型文件")
    args = parser.parse_args()

    try:
        if args.load:
            model = deserialize_model(args.load.read_bytes())
        else:
            model = train_bpe(TRAINING_DOCUMENTS)
        if args.save:
            with args.save.open("xb") as model_file:
                model_file.write(serialize_model(model))
        if args.csv:
            print(csv_text(model), end="")
        elif args.hand:
            print(hand_example())
        else:
            print(demo(model))
    except (OSError, TypeError, ValueError) as error:
        parser.exit(2, f"错误：{error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
