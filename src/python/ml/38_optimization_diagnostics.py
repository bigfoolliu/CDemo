"""比较小型二次目标上的优化器，并记录 MLP 训练诊断量。"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import math

INPUTS = ((-1.0, 0.0), (0.0, 1.0), (1.0, 0.0))
TARGETS = (0.0, 1.0, 0.0)
INITIAL_PARAMETERS = (0.2, -0.3, 0.1, 0.25, 0.1, -0.2, 0.05, 0.3, 0.0)


@dataclass
class OptimizerState:
    """保存二维目标所需的动量或 Adam 矩估计。"""

    velocity: list[float] = field(default_factory=lambda: [0.0, 0.0])
    first: list[float] = field(default_factory=lambda: [0.0, 0.0])
    second: list[float] = field(default_factory=lambda: [0.0, 0.0])
    step: int = 0


@dataclass(frozen=True)
class MlpMetrics:
    """保存损失与隐藏激活的简单统计。"""

    loss: float
    activation_minimum: float
    activation_maximum: float
    activation_mean: float


def quadratic_evaluate(point: tuple[float, float]) -> tuple[float, tuple[float, float]]:
    """计算 0.5(x²+9y²) 及其梯度。

    :param point: 两个有限参数坐标。
    :return: 标量损失与二元梯度。
    :raises ValueError: 参数或中间结果非有限时抛出。
    """
    if len(point) != 2 or any(not math.isfinite(value) for value in point):
        raise ValueError("目标需要两个有限参数")
    loss = 0.5 * (point[0] * point[0] + 9 * point[1] * point[1])
    gradient = (point[0], 9 * point[1])
    if not math.isfinite(loss) or any(not math.isfinite(v) for v in gradient):
        raise ValueError("目标计算非有限")
    return loss, gradient


def optimizer_step(point: tuple[float, float], gradient: tuple[float, float],
                   kind: str, rate: float, state: OptimizerState | None = None
                   ) -> tuple[tuple[float, float], OptimizerState | None]:
    """执行事务式 SGD、经典动量或 Adam 单步更新。

    :param point: 当前二元参数。
    :param gradient: 当前有限梯度。
    :param kind: sgd、momentum 或 adam。
    :param rate: 有限正学习率。
    :param state: 动量/Adam 状态；SGD 不使用。
    :return: 新参数与新优化器状态；SGD 状态仍为 None。
    :raises ValueError: 参数、状态、学习率或更新结果无效时抛出。
    """
    if (len(point) != 2 or len(gradient) != 2 or kind not in ("sgd", "momentum", "adam") or
            not math.isfinite(rate) or rate <= 0 or
            any(not math.isfinite(v) for v in (*point, *gradient)) or
            (kind != "sgd" and state is None)):
        raise ValueError("优化器输入无效")
    next_state = None if state is None else OptimizerState(
        state.velocity.copy(), state.first.copy(), state.second.copy(), state.step)
    if kind == "adam":
        assert next_state is not None
        next_state.step += 1
    updated = []
    for index, value in enumerate(point):
        direction = gradient[index]
        if kind == "momentum":
            assert next_state is not None
            next_state.velocity[index] = 0.5 * next_state.velocity[index] + gradient[index]
            direction = next_state.velocity[index]
        elif kind == "adam":
            assert next_state is not None
            next_state.first[index] = 0.9 * next_state.first[index] + 0.1 * gradient[index]
            next_state.second[index] = (0.999 * next_state.second[index] +
                                        0.001 * gradient[index] * gradient[index])
            first_hat = next_state.first[index] / (1 - 0.9 ** next_state.step)
            second_hat = next_state.second[index] / (1 - 0.999 ** next_state.step)
            direction = first_hat / (math.sqrt(second_hat) + 1e-8)
        result = value - rate * direction
        if not math.isfinite(result):
            raise ValueError("更新结果非有限；原参数与状态未改变")
        updated.append(result)
    return (updated[0], updated[1]), next_state


def mlp_evaluate(parameters: tuple[float, ...], with_gradient: bool = False
                 ) -> tuple[MlpMetrics, tuple[float, ...] | None]:
    """计算三样本两隐藏单元 tanh 网络的损失和可选梯度。

    :param parameters: 顺序为两组输入权重、隐藏偏置、输出权重和输出偏置的九参数元组。
    :param with_gradient: 为真时同时返回平均解析梯度。
    :return: 损失、激活范围/均值和可选的九维梯度。
    :raises ValueError: 参数形状或计算结果无效时抛出。
    """
    if len(parameters) != 9 or any(not math.isfinite(v) for v in parameters):
        raise ValueError("MLP 需要九个有限参数")
    loss = 0.0
    activations: list[float] = []
    gradients = [0.0] * 9
    for inputs, target in zip(INPUTS, TARGETS):
        hidden = []
        for unit in range(2):
            offset = 2 * unit
            z = (parameters[offset] * inputs[0] + parameters[offset + 1] * inputs[1] +
                 parameters[4 + unit])
            hidden.append(math.tanh(z))
        activations.extend(hidden)
        prediction = parameters[6] * hidden[0] + parameters[7] * hidden[1] + parameters[8]
        error = prediction - target
        loss += 0.5 * error * error / len(INPUTS)
        if with_gradient:
            delta = error / len(INPUTS)
            gradients[6] += delta * hidden[0]
            gradients[7] += delta * hidden[1]
            gradients[8] += delta
            for unit in range(2):
                offset = 2 * unit
                hidden_delta = delta * parameters[6 + unit] * (1 - hidden[unit] ** 2)
                gradients[offset] += hidden_delta * inputs[0]
                gradients[offset + 1] += hidden_delta * inputs[1]
                gradients[4 + unit] += hidden_delta
    metrics = MlpMetrics(loss, min(activations), max(activations),
                         sum(activations) / len(activations))
    if not math.isfinite(loss) or any(not math.isfinite(v) for v in activations):
        raise ValueError("MLP 前向结果非有限")
    if with_gradient:
        if any(not math.isfinite(v) for v in gradients):
            raise ValueError("MLP 梯度非有限")
        return metrics, tuple(gradients)
    return metrics, None


def mlp_train_step(parameters: tuple[float, ...], rate: float
                   ) -> tuple[tuple[float, ...], float, float]:
    """用一个完整批次执行 MLP 的 SGD 步，并测量范数。

    :param parameters: 当前九维网络参数。
    :param rate: 有限正学习率。
    :return: 新参数、梯度二范数和更新量二范数。
    :raises ValueError: 更新输入或结果无效时抛出。
    """
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError("学习率须为有限正数")
    _, gradients = mlp_evaluate(parameters, with_gradient=True)
    assert gradients is not None
    updated = tuple(value - rate * gradient
                    for value, gradient in zip(parameters, gradients))
    if any(not math.isfinite(value) for value in updated):
        raise ValueError("参数更新非有限")
    gradient_norm = math.sqrt(sum(value * value for value in gradients))
    update_norm = math.sqrt(sum((new - old) ** 2 for new, old in zip(updated, parameters)))
    return updated, gradient_norm, update_norm


def results() -> dict[str, float]:
    """计算三个优化器的相同预算轨迹和 MLP 固定诊断序列。

    :return: 带稳定字段名的损失、激活统计和更新范数。
    """
    values: dict[str, float] = {}
    for kind in ("sgd", "momentum", "adam"):
        point = (1.0, 1.0)
        state = None if kind == "sgd" else OptimizerState()
        for step in range(6):
            loss, gradient = quadratic_evaluate(point)
            values[f"{kind}_{step}"] = loss
            point, state = optimizer_step(point, gradient, kind, 0.1, state)
    parameters = INITIAL_PARAMETERS
    for step in range(41):
        metrics, _ = mlp_evaluate(parameters)
        values[f"mlp_loss_{step}"] = metrics.loss
        values[f"mlp_activation_mean_{step}"] = metrics.activation_mean
        values[f"mlp_activation_min_{step}"] = metrics.activation_minimum
        values[f"mlp_activation_max_{step}"] = metrics.activation_maximum
        if step < 40:
            parameters, grad_norm, update_norm = mlp_train_step(parameters, 0.1)
            values[f"mlp_gradient_norm_{step}"] = grad_norm
            values[f"mlp_update_norm_{step}"] = update_norm
        else:
            _, gradients = mlp_evaluate(parameters, with_gradient=True)
            assert gradients is not None
            values[f"mlp_gradient_norm_{step}"] = math.sqrt(
                sum(value * value for value in gradients))
            values[f"mlp_update_norm_{step}"] = 0.0
    return values


def main() -> None:
    """输出优化器轨迹及 MLP 诊断，或跨语言 CSV。

    :return: 无。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", action="store_true", help="输出 field,value")
    args = parser.parse_args()
    values = results()
    if args.csv:
        print("field,value")
        for field, value in values.items():
            print(f"{field},{value:.17g}")
        return
    for kind, label in (("sgd", "SGD"), ("momentum", "Momentum"), ("adam", "Adam")):
        for step in (0, 1, 5):
            print(f"{label:<8} 步={step}，损失={values[f'{kind}_{step}']:.6f}")
    for step in (0, 1, 2, 10, 20, 40):
        print("MLP 步={}，损失={:.6f}，激活均值={:.4f}，范围=[{:.4f},{:.4f}]，"
              "梯度范数={:.5f}，更新范数={:.5f}".format(
                  step, values[f"mlp_loss_{step}"], values[f"mlp_activation_mean_{step}"],
                  values[f"mlp_activation_min_{step}"], values[f"mlp_activation_max_{step}"],
                  values.get(f"mlp_gradient_norm_{step}", 0),
                  values.get(f"mlp_update_norm_{step}", 0)))


if __name__ == "__main__":
    main()
