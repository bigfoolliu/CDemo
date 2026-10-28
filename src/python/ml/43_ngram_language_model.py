"""用标准库实现固定小词表的一元和二元语言模型。

训练与测试数据按原始文档分开；BOS 仅作上下文，EOS 参与预测和评分。
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from typing import Sequence


VOCABULARY = (
    "<unk>",
    "<bos>",
    "<eos>",
    ".",
    "c",
    "i",
    "language",
    "learn",
    "like",
    "python",
    "study",
    "you",
)
TOKEN_TO_ID = {token: index for index, token in enumerate(VOCABULARY)}
UNK, BOS, EOS = 0, 1, 2
ALPHA = 1.0
MAX_SEQUENCE_LENGTH = 16
MAX_GENERATED_TOKENS = 12

# 每个字符串是一份原始文档；先拆分，再分别分词，避免窗口级数据泄漏。
TRAINING_TEXT = (
    "i like c .",
    "i like c .",
    "you like c .",
    "i learn c .",
    "you learn python .",
    "i like language .",
)
TESTING_TEXT = (
    "you like language .",
    "you study rust .",
)


@dataclass(frozen=True)
class Document:
    """保存一篇已经加入 BOS 和 EOS 的文档。"""

    tokens: tuple[int, ...]


@dataclass
class Model:
    """保存一元计数和二元转移计数。"""

    kind: str
    vocabulary_size: int
    unigram_count: list[int] = field(default_factory=list)
    bigram_count: list[list[int]] = field(default_factory=list)
    context_count: list[int] = field(default_factory=list)
    token_count: int = 0


@dataclass(frozen=True)
class Metrics:
    """保存按预测 token 平均的 NLL、困惑度和计分 token 数。"""

    nll: float
    perplexity: float
    token_count: int


def lookup_token(word: str) -> int:
    """查找固定词表编号，未知词统一映射到 UNK。

    :param word: 一个分词后的字符串。
    :return: 忽略 ASCII 大小写后的词表编号；词表外返回 UNK。
    """
    uppercase = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    lowercase = "abcdefghijklmnopqrstuvwxyz"
    normalized = word.translate(str.maketrans(uppercase, lowercase))
    return TOKEN_TO_ID.get(normalized, UNK)


def split_ascii_whitespace(text: str) -> tuple[str, ...]:
    """只按四种 ASCII 空白切分，保持 C/Python 规则一致。

    :param text: 已确认是字符串的输入文本。
    :return: 按四种 ASCII 空白拆分的非空 token。
    """
    words = []
    start = None
    for index, character in enumerate(text):
        if character in " \t\n\r":
            if start is not None:
                words.append(text[start:index])
                start = None
        elif start is None:
            start = index
    if start is not None:
        words.append(text[start:])
    return tuple(words)


def encode_document(text: str) -> Document:
    """使用空白分词，并在文档两端加入 BOS 与 EOS。

    :param text: 文档文本；仅空白分隔，不执行复杂标点切分。
    :return: 含起始符、词项和结束符的不可变 token 序列。
    :raises TypeError: text 不是字符串时抛出。
    :raises ValueError: 序列超过固定容量时抛出。
    """
    if not isinstance(text, str):
        raise TypeError("text 必须是字符串")
    tokens = (BOS, *(lookup_token(word) for word in split_ascii_whitespace(text)), EOS)
    if len(tokens) > MAX_SEQUENCE_LENGTH:
        raise ValueError("文档 token 数超过固定容量")
    return Document(tokens)


def encode_dataset(texts: Sequence[str]) -> tuple[Document, ...]:
    """逐篇编码数据集，不拆分或混合原始文档。

    :param texts: 原始文档序列。
    :return: 按输入顺序排列的编码文档。
    :raises ValueError: 数据集为空时抛出。
    """
    if not texts:
        raise ValueError("数据集至少需要一篇文档")
    return tuple(encode_document(text) for text in texts)


def train_model(documents: Sequence[Document], kind: str, vocabulary_size: int) -> Model:
    """只用传入的训练文档统计 unigram 或 bigram。

    :param documents: 训练文档；每篇必须以 BOS 开始、EOS 结束。
    :param kind: ``unigram`` 或 ``bigram``。
    :param vocabulary_size: 固定词表大小，须大于二且不超过 16。
    :return: 完成计数的模型。
    :raises ValueError: 模型类型、词表大小或文档 token 不合法时抛出。
    """
    if kind not in {"unigram", "bigram"}:
        raise ValueError("kind 必须是 unigram 或 bigram")
    if not 3 <= vocabulary_size <= 16:
        raise ValueError("vocabulary_size 必须位于 3..16")
    if not documents:
        raise ValueError("至少需要一篇训练文档")

    model = Model(
        kind=kind,
        vocabulary_size=vocabulary_size,
        unigram_count=[0] * vocabulary_size,
        bigram_count=[[0] * vocabulary_size for _ in range(vocabulary_size)],
        context_count=[0] * vocabulary_size,
    )
    for document in documents:
        tokens = document.tokens
        if len(tokens) < 2 or tokens[0] != BOS or tokens[-1] != EOS:
            raise ValueError("文档必须以 BOS 开始、EOS 结束")
        for history, target in zip(tokens, tokens[1:]):
            if not 0 <= history < vocabulary_size or not 0 <= target < vocabulary_size:
                raise ValueError("token 编号超出词表")
            if target == BOS:
                raise ValueError("BOS 只能作为上下文，不能作为预测目标")
            model.unigram_count[target] += 1
            model.bigram_count[history][target] += 1
            model.context_count[history] += 1
            model.token_count += 1
    return model


def next_probabilities(model: Model, history: int = BOS) -> tuple[float, ...]:
    """计算加性平滑后的下一 token 概率行。

    :param model: 已训练模型。
    :param history: 二元模型上下文；一元模型忽略此值。
    :return: 词表顺序的概率元组，BOS 对应概率为零。
    :raises ValueError: 上下文编号或模型类型不合法时抛出。
    """
    if not 0 <= history < model.vocabulary_size:
        raise ValueError("history 超出词表")
    if model.kind not in {"unigram", "bigram"}:
        raise ValueError("未知模型类型")
    outcome_count = model.vocabulary_size - 1
    observed = model.token_count if model.kind == "unigram" else model.context_count[history]
    denominator = observed + ALPHA * outcome_count
    probabilities = []
    for token in range(model.vocabulary_size):
        if token == BOS:
            probabilities.append(0.0)
            continue
        count = (
            model.unigram_count[token]
            if model.kind == "unigram"
            else model.bigram_count[history][token]
        )
        probabilities.append((count + ALPHA) / denominator)
    return tuple(probabilities)


def evaluate(model: Model, documents: Sequence[Document]) -> Metrics:
    """计算按词项和 EOS 计分的平均 NLL 与困惑度。

    :param model: 已训练的一元或二元模型。
    :param documents: 独立评估文档，不参与模型计数。
    :return: token 平均 NLL、困惑度及目标 token 总数。
    :raises ValueError: 没有预测目标或目标超出模型词表时抛出。
    """
    total_nll = 0.0
    total_tokens = 0
    for document in documents:
        tokens = document.tokens
        if len(tokens) < 2 or tokens[0] != BOS or tokens[-1] != EOS:
            raise ValueError("评估文档必须以 BOS 开始、EOS 结束")
        for history, target in zip(tokens, tokens[1:]):
            if target == BOS or target >= model.vocabulary_size:
                raise ValueError("评估目标不属于可预测词表")
            probability = next_probabilities(model, history)[target]
            if not probability > 0.0:
                raise ValueError("目标 token 概率必须大于零")
            total_nll -= math.log(probability)
            total_tokens += 1
    if total_tokens == 0:
        raise ValueError("评估集没有可计分 token")
    average_nll = total_nll / total_tokens
    return Metrics(average_nll, math.exp(average_nll), total_tokens)


def sample(model: Model, seed: int, capacity: int = MAX_GENERATED_TOKENS) -> tuple[int, ...]:
    """按固定 LCG 随机采样，遇到 EOS 或长度上限时停止。

    :param model: 已训练模型。
    :param seed: 可复现种子，不用于安全用途。
    :param capacity: 最大生成 token 数，范围为 1..12。
    :return: 生成 token 序列，EOS 如被生成也包含在内。
    :raises ValueError: 容量或种子不合法时抛出。
    """
    if not 1 <= capacity <= MAX_GENERATED_TOKENS or seed < 0:
        raise ValueError("采样容量或随机种子不合法")
    state = seed & 0xFFFFFFFF
    history = BOS
    generated = []
    for _ in range(capacity):
        probabilities = next_probabilities(model, history)
        state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
        choice = (state >> 8) / 16777216.0
        cumulative = 0.0
        selected = EOS
        for token, probability in enumerate(probabilities):
            cumulative += probability
            if token != BOS and choice < cumulative:
                selected = token
                break
        generated.append(selected)
        if selected == EOS:
            break
        history = selected
    return tuple(generated)


def hand_example() -> dict[str, tuple[float, ...]]:
    """生成与 C 案例相同的小型手算夹具。

    :return: 概率行和句子概率、平均 NLL、困惑度组成的数值字典。
    """
    tiny_document = Document((BOS, 3, 4, EOS))
    unigram = train_model((tiny_document,), "unigram", 5)
    bigram = train_model((tiny_document,), "bigram", 5)
    uni_row = next_probabilities(unigram)
    bos_row = next_probabilities(bigram, BOS)
    a_row = next_probabilities(bigram, 3)
    b_row = next_probabilities(bigram, 4)
    uni_sentence = uni_row[3] * uni_row[4] * uni_row[EOS]
    bi_sentence = bos_row[3] * a_row[4] * b_row[EOS]
    uni_nll = -math.log(uni_sentence) / 3
    bi_nll = -math.log(bi_sentence) / 3
    return {
        "unigram": (
            uni_row[3], uni_row[4], uni_row[EOS], uni_row[UNK],
            uni_sentence, uni_nll, math.exp(uni_nll),
        ),
        "bigram": (
            bos_row[3], a_row[4], b_row[EOS],
            bi_sentence, bi_nll, math.exp(bi_nll),
        ),
    }


def csv_text() -> str:
    """返回与 C ``--csv`` 命令字段顺序相同的数值行。

    :return: 包含两条测试指标和两条手算行的 CSV 文本，不含标题行。
    """
    training = encode_dataset(TRAINING_TEXT)
    testing = encode_dataset(TESTING_TEXT)
    rows = []
    for kind in ("unigram", "bigram"):
        metrics = evaluate(train_model(training, kind, len(VOCABULARY)), testing)
        rows.append(
            f"test,{kind},{metrics.token_count},{metrics.nll:.12g},{metrics.perplexity:.12g}"
        )
    values = hand_example()
    for kind in ("unigram", "bigram"):
        numbers = ",".join(f"{value:.12g}" for value in values[kind])
        rows.append(f"hand,{kind},{numbers}")
    return "\n".join(rows) + "\n"


def run_csv() -> None:
    """输出机器可读的留出指标和手算中间值。

    :return: 无。
    """
    print(csv_text(), end="")


def main(arguments: Sequence[str] | None = None) -> int:
    """运行演示、手算输出或测试。

    :param arguments: 命令行参数；省略时读取 sys.argv。
    :return: 进程退出码，成功为 0。
    """
    args = list(sys.argv[1:] if arguments is None else arguments)
    if args == ["--csv"]:
        run_csv()
        return 0
    if args == ["--hand"]:
        values = hand_example()
        uni = values["unigram"]
        bi = values["bigram"]
        print(
            "手算语料：BOS a b EOS；可预测词表为 UNK、EOS、a、b，"
            "V_next=4，alpha=1"
        )
        print(
            "一元计数：a=1、b=1、EOS=1、UNK=0、N=3；"
            "平滑概率分别为 2/7、2/7、2/7、1/7"
        )
        print(
            f"一元句子概率：P(a b EOS)=(2/7)^3={uni[4]:.9f}，"
            f"token NLL={uni[5]:.9f}，PPL={uni[6]:.9f}"
        )
        print("二元转移：BOS→a=2/5，a→b=2/5，b→EOS=2/5")
        print(
            f"二元句子概率：P(a b EOS|BOS)=(2/5)^3={bi[3]:.9f}，"
            f"token NLL={bi[4]:.9f}，PPL={bi[5]:.9f}"
        )
        return 0
    if args:
        print("用法：43_ngram_language_model.py [--csv|--hand]", file=sys.stderr)
        return 2

    training = encode_dataset(TRAINING_TEXT)
    testing = encode_dataset(TESTING_TEXT)
    print(
        f"按原始文档拆分：训练 {len(training)} 篇，测试 {len(testing)} 篇；"
        f"词表 {len(VOCABULARY)} 个，UNK 包含 OOV"
    )
    print(f"评分单位：词项加 EOS；BOS 仅作上下文。加性平滑 alpha={ALPHA:.1f}")
    print("模型    测试 token 数    token 平均 NLL    困惑度 PPL")
    models = {}
    for kind in ("unigram", "bigram"):
        model = train_model(training, kind, len(VOCABULARY))
        metrics = evaluate(model, testing)
        models[kind] = model
        print(f"{kind:7s} {metrics.token_count:12d} {metrics.nll:17.6f} {metrics.perplexity:13.6f}")
        generated = sample(model, 2026)
        text = " ".join(VOCABULARY[token] for token in generated if token != EOS)
        truncated = len(generated) == MAX_GENERATED_TOKENS and generated[-1] != EOS
        suffix = " …[达到长度上限]" if truncated else ""
        print(f"{kind} 生成: {text}{suffix}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
