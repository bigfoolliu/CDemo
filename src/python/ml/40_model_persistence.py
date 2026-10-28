"""保存训练集标准化参数和逻辑回归权重，并在独立进程中加载推理。"""

import csv
from dataclasses import dataclass
import math
from pathlib import Path
import struct
import sys
import zlib

FEATURE_COUNT = 2
FORMAT_VERSION = 1
PARAMETER_COUNT = 7
HEADER = struct.Struct("<8sIIII")
PARAMETERS = struct.Struct("<7d")
CHECKSUM = struct.Struct("<I")
MAGIC = b"CDMLPRST"
MODEL_BYTES = HEADER.size + PARAMETERS.size + CHECKSUM.size
DEFAULT_CSV = Path("data/ml/study_outcomes.csv")
LEARNING_RATE = 0.15
TRAINING_EPOCHS = 1200


@dataclass(frozen=True)
class PersistedModel:
    """仅包含预测必需的预处理统计量和逻辑回归参数。"""

    mean: tuple[float, float]
    scale: tuple[float, float]
    weights: tuple[float, float]
    bias: float


def model_is_valid(model: PersistedModel) -> bool:
    """检查均值、标准差、权重和偏置是否可用于推理。

    :param model: 待验证的不可变推理状态。
    :return: 所有参数有限且标准差大于机器精度时为 True。
    """
    if len(model.mean) != FEATURE_COUNT or len(model.scale) != FEATURE_COUNT or \
            len(model.weights) != FEATURE_COUNT:
        return False
    values = (*model.mean, *model.scale, *model.weights, model.bias)
    return all(math.isfinite(value) for value in values) and all(
        value > sys.float_info.epsilon for value in model.scale)


def encode_model(model: PersistedModel) -> bytes:
    """将推理状态编码为带版本、维度和 CRC-32 的小端序文件。

    :param model: 要持久化的推理状态。
    :return: 固定长度的模型文件字节。
    :raises ValueError: 模型含无效参数时抛出。
    """
    if not model_is_valid(model):
        raise ValueError("模型参数无效")
    header = HEADER.pack(MAGIC, FORMAT_VERSION, FEATURE_COUNT, PARAMETER_COUNT,
                         PARAMETERS.size)
    payload = PARAMETERS.pack(*model.mean, *model.scale, *model.weights, model.bias)
    body = header + payload
    return body + CHECKSUM.pack(zlib.crc32(body))


def decode_model(data: bytes) -> PersistedModel:
    """严格验证模型文件后构造不可变状态，不暴露部分解析结果。

    :param data: 完整模型文件字节。
    :return: 校验通过的推理状态。
    :raises ValueError: 长度、魔数、版本、维度、校验和或参数无效时抛出。
    """
    if len(data) != MODEL_BYTES:
        raise ValueError("模型文件长度错误")
    magic, version, feature_count, parameter_count, payload_bytes = HEADER.unpack_from(data)
    if magic != MAGIC:
        raise ValueError("模型魔数错误")
    if version != FORMAT_VERSION:
        raise ValueError("不支持的模型版本")
    if feature_count != FEATURE_COUNT or parameter_count != PARAMETER_COUNT or \
            payload_bytes != PARAMETERS.size:
        raise ValueError("模型维度或参数数量错误")
    expected_crc, = CHECKSUM.unpack_from(data, len(data) - CHECKSUM.size)
    if expected_crc != zlib.crc32(data[:-CHECKSUM.size]):
        raise ValueError("模型校验和错误")

    values = PARAMETERS.unpack_from(data, HEADER.size)
    model = PersistedModel(
        mean=(values[0], values[1]),
        scale=(values[2], values[3]),
        weights=(values[4], values[5]),
        bias=values[6],
    )
    if not model_is_valid(model):
        raise ValueError("模型参数含非有限值或无效尺度")
    return model


def save_model_exclusive(path: Path, model: PersistedModel) -> None:
    """只创建新模型文件，拒绝覆盖已有用户模型。

    :param path: 目标文件路径；父目录必须已存在。
    :param model: 要保存的推理状态。
    :return: 无。
    :raises FileExistsError: 目标文件已经存在时抛出。
    :raises OSError: 路径不可写或写入失败时抛出。
    :raises ValueError: 模型参数无效时抛出。
    """
    encoded = encode_model(model)
    created = False
    try:
        with path.open("xb") as stream:
            created = True
            stream.write(encoded)
            stream.flush()
    except OSError:
        if created:
            path.unlink(missing_ok=True)
        raise


def load_model(path: Path) -> PersistedModel:
    """读取整个文件并在全部验证成功后返回推理状态。

    :param path: 模型文件路径。
    :return: 校验通过的不可变推理状态。
    :raises OSError: 文件无法读取时抛出。
    :raises ValueError: 模型文件格式或参数无效时抛出。
    """
    return decode_model(path.read_bytes())


def sigmoid(value: float) -> float:
    """稳定计算逻辑函数。

    :param value: 有限 logit。
    :return: [0, 1] 范围内的概率。
    """
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def binary_loss_gradient(logit: float, label: int) -> tuple[float, float]:
    """计算稳定二元交叉熵和对 logit 的解析导数。

    :param logit: 有限线性分数。
    :param label: 真实二分类标签 0 或 1。
    :return: 单样本损失及 logit 梯度。
    :raises ValueError: 分数非有限或标签不属于 0/1 时抛出。
    """
    if not math.isfinite(logit) or label not in (0, 1):
        raise ValueError("logit 或二分类标签无效")
    loss = max(logit, 0.0) - label * logit + math.log1p(math.exp(-abs(logit)))
    gradient = sigmoid(logit) - label
    if not math.isfinite(loss) or not math.isfinite(gradient):
        raise ValueError("损失或梯度非有限")
    return loss, gradient


def predict(model: PersistedModel, features: tuple[float, float]) -> tuple[
        tuple[float, float], float, float, int]:
    """使用训练时的统计量将原始特征映射为概率和类别。

    :param model: 已加载且有效的推理状态。
    :param features: 两个有限原始特征。
    :return: 标准化特征、logit、正类概率和 0/1 类别。
    :raises ValueError: 模型无效、特征维度错误或中间值非有限时抛出。
    """
    if not model_is_valid(model) or len(features) != FEATURE_COUNT or not all(
            math.isfinite(value) for value in features):
        raise ValueError("模型或输入特征无效")
    standardized = tuple((features[index] - model.mean[index]) / model.scale[index]
                         for index in range(FEATURE_COUNT))
    logit = model.bias + sum(model.weights[index] * standardized[index]
                             for index in range(FEATURE_COUNT))
    probability = sigmoid(logit)
    if not all(math.isfinite(value) for value in (*standardized, logit, probability)):
        raise ValueError("推理中间值非有限")
    return standardized, logit, probability, int(probability >= 0.5)


def train_from_csv(path: Path = DEFAULT_CSV) -> PersistedModel:
    """仅用 CSV 训练划分拟合标准化和固定轮数的逻辑回归。

    :param path: 含 train/validation/test 划分的 CSV 文件。
    :return: 用于独立推理的标准化统计量和最佳可用参数。
    :raises ValueError: 数据缺少特征、训练行或有限变化时抛出。
    :raises OSError: CSV 无法读取时抛出。
    """
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    train = [row for row in rows if row["split"] == "train"]
    if len(train) < 2:
        raise ValueError("训练集样本不足")

    feature_names = ("study_hours", "practice_questions")
    raw = [tuple(float(row[name]) for name in feature_names) for row in train]
    if not all(math.isfinite(value) for row in raw for value in row):
        raise ValueError("训练特征含非有限值")
    mean = tuple(sum(row[index] for row in raw) / len(raw)
                 for index in range(FEATURE_COUNT))
    scale = tuple(math.sqrt(sum((row[index] - mean[index]) ** 2 for row in raw) / len(raw))
                  for index in range(FEATURE_COUNT))
    if not all(math.isfinite(value) and value > sys.float_info.epsilon for value in scale):
        raise ValueError("训练集特征必须有有限的非零方差")

    normalized = [tuple((row[index] - mean[index]) / scale[index]
                        for index in range(FEATURE_COUNT)) for row in raw]
    labels = [int(row["passed"]) for row in train]
    if set(labels) != {0, 1}:
        raise ValueError("训练标签必须同时包含 0 和 1")

    weights = [0.0, 0.0]
    bias = 0.0
    for _ in range(TRAINING_EPOCHS):
        weight_gradient = [0.0, 0.0]
        bias_gradient = 0.0
        training_loss = 0.0
        for features, label in zip(normalized, labels):
            logit = bias + sum(weights[index] * features[index]
                               for index in range(FEATURE_COUNT))
            sample_loss, error = binary_loss_gradient(logit, label)
            training_loss += sample_loss
            for index in range(FEATURE_COUNT):
                weight_gradient[index] += error * features[index]
            bias_gradient += error
        if not math.isfinite(training_loss):
            raise ValueError("训练损失非有限")
        weights = [weights[index] - LEARNING_RATE * weight_gradient[index] / len(train)
                   for index in range(FEATURE_COUNT)]
        bias -= LEARNING_RATE * bias_gradient / len(train)

    model = PersistedModel(mean, scale, tuple(weights), bias)
    if not model_is_valid(model):
        raise ValueError("训练得到的模型参数无效")
    return model


def format_prediction(features: tuple[float, float], result: tuple[
        tuple[float, float], float, float, int]) -> str:
    """以稳定字段顺序格式化预测，供命令行和跨语言测试使用。

    :param features: 原始输入特征。
    :param result: predict 返回的标准化值、分数、概率和类别。
    :return: 单行预测文本。
    """
    standardized, logit, probability, label = result
    return (f"features={features[0]:.17g},{features[1]:.17g} "
            f"standardized={standardized[0]:.17g},{standardized[1]:.17g} "
            f"logit={logit:.17g} probability={probability:.17g} class={label}")


def main(arguments: list[str] | None = None) -> int:
    """运行演示、独占保存训练结果或独立加载预测。

    :param arguments: 可选命令行参数；省略时读取 sys.argv。
    :return: 进程退出状态码。
    """
    args = list(sys.argv[1:] if arguments is None else arguments)
    try:
        if not args or args == ["demo"]:
            model = train_from_csv()
            features = (5.2, 64.0)
            print("训练 -> 序列化 -> 解码 -> 对新样本预测：")
            print("train_mean={} train_scale={} weights={} bias={:.6f}".format(
                ",".join(f"{value:.6f}" for value in model.mean),
                ",".join(f"{value:.6f}" for value in model.scale),
                ",".join(f"{value:.6f}" for value in model.weights), model.bias))
            print(format_prediction(features, predict(decode_model(encode_model(model)), features)))
            print("持久文件：python3 src/python/ml/40_model_persistence.py train PATH")
            print("独立推理：python3 src/python/ml/40_model_persistence.py predict PATH 5.2 64")
            return 0
        if len(args) == 2 and args[0] == "train":
            model = train_from_csv()
            save_model_exclusive(Path(args[1]), model)
            print(f"已保存版本 {FORMAT_VERSION} 模型：{args[1]}")
            return 0
        if len(args) == 4 and args[0] == "predict":
            model = load_model(Path(args[1]))
            features = (float(args[2]), float(args[3]))
            print(format_prediction(features, predict(model, features)))
            return 0
    except (OSError, ValueError, OverflowError, struct.error) as error:
        print(f"模型操作失败：{error}", file=sys.stderr)
        return 1

    print("用法：[demo | train MODEL_PATH | predict MODEL_PATH FEATURE_0 FEATURE_1]",
          file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
