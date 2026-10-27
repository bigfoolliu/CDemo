"""M02：用两条二维样本推导批量全连接层的三类梯度。"""

from __future__ import annotations

import argparse
import math
from typing import NamedTuple


class Model(NamedTuple):
    """行是输入特征、列是输出神经元的全连接层参数。"""

    weights: tuple[tuple[float, float], tuple[float, float]]
    bias: tuple[float, float]


class Batch(NamedTuple):
    """批次样本及目标；每行都有两个元素。"""

    inputs: tuple[tuple[float, float], ...]
    targets: tuple[tuple[float, float], ...]


class Forward(NamedTuple):
    """同一组旧参数下的预测、误差及平均半平方损失。"""

    predictions: tuple[tuple[float, float], ...]
    errors: tuple[tuple[float, float], ...]
    loss: float


class Gradients(NamedTuple):
    """对输入、权重和偏置的梯度。"""

    inputs: tuple[tuple[float, float], ...]
    weights: tuple[tuple[float, float], tuple[float, float]]
    bias: tuple[float, float]


INITIAL_MODEL = Model(((1.0, -1.0), (2.0, 0.5)), (0.5, -0.5))
INITIAL_BATCH = Batch(((1.0, 2.0), (-1.0, 1.0)), ((2.0, 0.0), (0.0, 1.0)))


def validate(model: Model, batch: Batch) -> None:
    """检查固定二维形状、批次范围与有限数。

    :param model: 待检查的全连接层参数。
    :param batch: 待检查的样本和目标。
    :return: 无。
    :raises ValueError: 形状、批大小或数值不符合教学示例约定。
    """
    if len(model.weights) != 2 or any(len(row) != 2 for row in model.weights):
        raise ValueError("权重必须是 2×2 矩阵")
    if len(model.bias) != 2 or not 1 <= len(batch.inputs) <= 3:
        raise ValueError("偏置宽度或批大小不符合约定")
    if len(batch.inputs) != len(batch.targets):
        raise ValueError("样本和目标数量不同")
    if any(len(row) != 2 for row in batch.inputs + batch.targets):
        raise ValueError("样本和目标都必须有两个元素")
    if not all(math.isfinite(value) for row in model.weights for value in row):
        raise ValueError("权重必须有限")
    if not all(math.isfinite(value) for value in model.bias):
        raise ValueError("偏置必须有限")
    if not all(math.isfinite(value) for row in batch.inputs + batch.targets for value in row):
        raise ValueError("样本和目标必须有限")


def dense_forward(model: Model, batch: Batch) -> Forward:
    """计算预测、误差和按批次及输出宽度平均的半平方损失。

    :param model: 固定二维的只读参数。
    :param batch: 一至三条只读样本及目标。
    :return: 同一参数版本下的前向缓存。
    :raises ValueError: 输入或中间结果非法。
    """
    validate(model, batch)
    predictions = []
    errors = []
    squared_sum = 0.0
    for sample, target in zip(batch.inputs, batch.targets):
        row_predictions = []
        row_errors = []
        for output in range(2):
            prediction = model.bias[output]
            for feature in range(2):
                prediction += sample[feature] * model.weights[feature][output]
            error = prediction - target[output]
            squared = error * error
            if not all(map(math.isfinite, (prediction, error, squared))):
                raise ValueError("前向结果必须有限")
            row_predictions.append(prediction)
            row_errors.append(error)
            squared_sum += squared
            if not math.isfinite(squared_sum):
                raise ValueError("损失求和溢出")
        predictions.append(tuple(row_predictions))
        errors.append(tuple(row_errors))
    loss = squared_sum / (2.0 * len(batch.inputs) * 2.0)
    return Forward(tuple(predictions), tuple(errors), loss)


def dense_backward(model: Model, batch: Batch, forward: Forward) -> Gradients:
    """用同一参数版本的前向缓存计算三类梯度。

    :param model: 生成 forward 时使用的旧参数。
    :param batch: 生成 forward 时使用的旧批次。
    :param forward: 对应批次和旧参数的前向缓存，调用者保证未过期。
    :return: 对输入、权重和偏置的有限梯度。
    :raises ValueError: 输入或梯度非法。
    """
    validate(model, batch)
    if (len(forward.errors) != len(batch.inputs) or
            any(len(row) != 2 for row in forward.errors) or
            not math.isfinite(forward.loss)):
        raise ValueError("前向缓存形状或损失非法")
    input_gradients = [[0.0, 0.0] for _ in batch.inputs]
    weight_gradients = [[0.0, 0.0], [0.0, 0.0]]
    bias_gradients = [0.0, 0.0]
    denominator = float(len(batch.inputs) * 2)
    for sample in range(len(batch.inputs)):
        for output in range(2):
            error = forward.errors[sample][output]
            if not math.isfinite(error):
                raise ValueError("前向误差必须有限")
            delta = error / denominator
            bias_gradients[output] += delta
            for feature in range(2):
                weight_gradients[feature][output] += batch.inputs[sample][feature] * delta
                input_gradients[sample][feature] += model.weights[feature][output] * delta
    all_values = (value for matrix in (input_gradients, weight_gradients)
                  for row in matrix for value in row)
    if not all(map(math.isfinite, all_values)) or not all(map(math.isfinite, bias_gradients)):
        raise ValueError("梯度必须有限")
    return Gradients(tuple(tuple(row) for row in input_gradients),
                     (tuple(weight_gradients[0]), tuple(weight_gradients[1])),
                     tuple(bias_gradients))


def dense_update(model: Model, gradients: Gradients, rate: float) -> Model:
    """用同一版本参数的梯度更新权重和偏置。

    :param model: 原参数；此函数不修改它。
    :param gradients: 对权重和偏置的有限梯度。
    :param rate: 有限正学习率。
    :return: 更新后的新参数。
    :raises ValueError: 学习率、梯度或结果非法。
    """
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError("学习率必须为有限正数")
    weights = []
    for feature in range(2):
        row = []
        for output in range(2):
            gradient = gradients.weights[feature][output]
            if not math.isfinite(gradient):
                raise ValueError("权重梯度必须有限")
            value = model.weights[feature][output] - rate * gradient
            if not math.isfinite(value):
                raise ValueError("更新后的权重必须有限")
            row.append(value)
        weights.append(tuple(row))
    bias = []
    for output in range(2):
        gradient = gradients.bias[output]
        if not math.isfinite(gradient):
            raise ValueError("偏置梯度必须有限")
        value = model.bias[output] - rate * gradient
        if not math.isfinite(value):
            raise ValueError("更新后的偏置必须有限")
        bias.append(value)
    return Model((weights[0], weights[1]), tuple(bias))


def result_rows() -> list[tuple[str, float]]:
    """生成与 C 版本同顺序、同数值的示例输出。

    :return: 字段名称与有限值列表。
    """
    forward = dense_forward(INITIAL_MODEL, INITIAL_BATCH)
    gradients = dense_backward(INITIAL_MODEL, INITIAL_BATCH, forward)
    updated = dense_update(INITIAL_MODEL, gradients, 0.1)
    after = dense_forward(updated, INITIAL_BATCH)
    rows = []
    for sample in range(2):
        for output in range(2):
            rows.append((f"z_{sample}_{output}", forward.predictions[sample][output]))
    rows.append(("loss", forward.loss))
    for sample in range(2):
        for feature in range(2):
            rows.append((f"dx_{sample}_{feature}", gradients.inputs[sample][feature]))
    for feature in range(2):
        for output in range(2):
            rows.append((f"dw_{feature}_{output}", gradients.weights[feature][output]))
            rows.append((f"w_next_{feature}_{output}", updated.weights[feature][output]))
    for output in range(2):
        rows.append((f"db_{output}", gradients.bias[output]))
        rows.append((f"b_next_{output}", updated.bias[output]))
    rows.append(("loss_next", after.loss))
    return rows


def main() -> None:
    """运行无随机数、无文件写入的教学示例。

    :return: 无。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", action="store_true", help="输出 C/Python 对照 CSV")
    args = parser.parse_args()
    if args.csv:
        print("field,value")
    for label, value in result_rows():
        if args.csv:
            print(f"{label},{value:.17g}")
        else:
            print(f"{label:<14} = {value:.10f}")


if __name__ == "__main__":
    main()
