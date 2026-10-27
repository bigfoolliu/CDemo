"""用标量计算图学习反向自动微分和 XOR 训练。"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

CAPACITY = 128
INITIAL = (2.0, 2.0, -1.0, 2.0, 2.0, -3.0, 2.0, -2.0, -1.0)
SAMPLES = ((0.0, 0.0, 0.0), (0.0, 1.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 0.0))
EPOCHS = 800


@dataclass
class Node:
    """保存一个标量的前向值、父节点索引及反向梯度。"""

    op: str
    left: int | None
    right: int | None
    value: float
    gradient: float = 0.0


class Graph:
    """固定容量计算图；节点只引用先创建的节点。"""

    def __init__(self, capacity: int = CAPACITY) -> None:
        """创建空图。

        :param capacity: 最大节点数，范围为 1 到 128。
        :return: 无。
        """
        if not 1 <= capacity <= CAPACITY:
            raise ValueError("容量须在 1 到 128 之间")
        self.capacity = capacity
        self.nodes: list[Node] = []

    def reset(self) -> None:
        """丢弃旧图和梯度，旧索引不再有效。

        :return: 无。
        """
        self.nodes.clear()

    def push(self, op: str, left: int | None, right: int | None, value: float) -> int:
        """追加有限值节点，失败时不修改图。

        :param op: 运算名称。
        :param left: 左父节点索引；叶节点为 ``None``。
        :param right: 右父节点索引；单目或叶节点为 ``None``。
        :param value: 已计算的前向标量。
        :return: 新节点索引。
        """
        if op not in {"leaf", "add", "mul", "exp", "log", "tanh", "sigmoid"}:
            raise ValueError("未知运算")
        if len(self.nodes) >= self.capacity or not math.isfinite(value):
            raise ValueError("容量已满或数值非有限")
        if op == "leaf" and (left is not None or right is not None):
            raise ValueError("叶节点不能有父节点")
        if op != "leaf" and (left is None or not 0 <= left < len(self.nodes)):
            raise ValueError("左父节点无效")
        if op in ("add", "mul") and (right is None or not 0 <= right < len(self.nodes)):
            raise ValueError("右父节点无效")
        index = len(self.nodes)
        self.nodes.append(Node(op, left, right, value))
        return index

    def leaf(self, value: float) -> int:
        """加入变量或常数。

        :param value: 有限的前向值。
        :return: 叶节点索引。
        """
        return self.push("leaf", None, None, value)

    def add(self, left: int, right: int) -> int:
        """创建加法节点。

        :param left: 左输入索引。
        :param right: 右输入索引，可与左输入相同。
        :return: 新节点索引。
        """
        if not 0 <= left < len(self.nodes) or not 0 <= right < len(self.nodes):
            raise ValueError("父节点无效")
        return self.push("add", left, right,
                         self.nodes[left].value + self.nodes[right].value)

    def mul(self, left: int, right: int) -> int:
        """创建乘法节点。

        :param left: 左输入索引。
        :param right: 右输入索引，可与左输入相同。
        :return: 新节点索引。
        """
        if not 0 <= left < len(self.nodes) or not 0 <= right < len(self.nodes):
            raise ValueError("父节点无效")
        return self.push("mul", left, right,
                         self.nodes[left].value * self.nodes[right].value)

    def unary(self, op: str, index: int) -> int:
        """创建指数、对数或激活节点。

        :param op: ``exp``、``log``、``tanh`` 或 ``sigmoid``。
        :param index: 输入节点索引。
        :return: 新节点索引。
        """
        if not 0 <= index < len(self.nodes):
            raise ValueError("输入节点无效")
        x = self.nodes[index].value
        if op == "exp":
            try:
                value = math.exp(x)
            except OverflowError as exc:
                raise ValueError("指数结果溢出") from exc
        elif op == "log":
            if x <= 0:
                raise ValueError("对数输入必须为正")
            value = math.log(x)
        elif op == "tanh":
            value = math.tanh(x)
        elif op == "sigmoid":
            value = 1 / (1 + math.exp(-x)) if x >= 0 else math.exp(x) / (1 + math.exp(x))
        else:
            raise ValueError("未知运算")
        return self.push(op, index, None, value)

    def backward(self, output: int) -> None:
        """从输出逆序传播；每次调用先清除旧梯度。

        :param output: 有效输出索引。
        :return: 无；非有限梯度时抛出异常并清零。
        """
        if not 0 <= output < len(self.nodes):
            raise ValueError("输出节点无效")
        for node in self.nodes:
            node.gradient = 0.0
        self.nodes[output].gradient = 1.0
        try:
            for node in reversed(self.nodes[:output + 1]):
                if node.op == "leaf":
                    continue
                assert node.left is not None
                left = self.nodes[node.left]
                right = self.nodes[node.right] if node.right is not None else None
                upstream = node.gradient
                if node.op == "add":
                    left_delta = right_delta = upstream
                elif node.op == "mul":
                    assert node.right is not None
                    assert right is not None
                    left_delta = upstream * right.value
                    right_delta = upstream * left.value
                elif node.op == "exp":
                    left_delta = upstream * node.value
                elif node.op == "log":
                    left_delta = upstream / left.value
                elif node.op == "tanh":
                    left_delta = upstream * (1 - node.value * node.value)
                elif node.op == "sigmoid":
                    left_delta = upstream * node.value * (1 - node.value)
                else:
                    raise ValueError("未知运算")
                left.gradient += left_delta
                if node.op in ("add", "mul"):
                    assert right is not None
                    right.gradient += right_delta
                if not math.isfinite(left.gradient):
                    raise ValueError("梯度非有限")
                if right is not None and not math.isfinite(right.gradient):
                    raise ValueError("梯度非有限")
        except (ValueError, OverflowError):
            for node in self.nodes:
                node.gradient = 0.0
            raise


def affine2(graph: Graph, w0: int, w1: int, bias: int, x0: int, x1: int) -> int:
    """构造两个输入的仿射变换。

    :param graph: 当前计算图。
    :param w0: 第一个权重节点。
    :param w1: 第二个权重节点。
    :param bias: 偏置节点。
    :param x0: 第一个输入节点。
    :param x1: 第二个输入节点。
    :return: 仿射输出节点索引。
    """
    return graph.add(graph.add(graph.mul(w0, x0), graph.mul(w1, x1)), bias)


def xor_sample(parameters: tuple[float, ...], x0: float, x1: float,
               target: float) -> tuple[float, float, tuple[float, ...]]:
    """为单条 XOR 数据建图并返回预测、损失及九个参数梯度。

    :param parameters: 九个有限参数，按两组隐藏层和一组输出层排列。
    :param x0: 第一个二值输入。
    :param x1: 第二个二值输入。
    :param target: 二值目标。
    :return: 预测概率、半平方损失和梯度元组。
    """
    if len(parameters) != 9 or any(not math.isfinite(p) for p in parameters):
        raise ValueError("参数必须为九个有限数")
    if x0 not in (0, 1) or x1 not in (0, 1) or target not in (0, 1):
        raise ValueError("XOR 输入和目标必须为二值")
    graph = Graph()
    leaves = [graph.leaf(p) for p in parameters]
    input0, input1 = graph.leaf(x0), graph.leaf(x1)
    target_node, minus_one, half = graph.leaf(target), graph.leaf(-1), graph.leaf(0.5)
    hidden1 = graph.unary("tanh", affine2(graph, *leaves[:3], input0, input1))
    hidden2 = graph.unary("tanh", affine2(graph, *leaves[3:6], input0, input1))
    logit = affine2(graph, *leaves[6:9], hidden1, hidden2)
    prediction_node = graph.unary("sigmoid", logit)
    error = graph.add(prediction_node, graph.mul(target_node, minus_one))
    loss_node = graph.mul(half, graph.mul(error, error))
    graph.backward(loss_node)
    return (graph.nodes[prediction_node].value, graph.nodes[loss_node].value,
            tuple(graph.nodes[i].gradient for i in leaves))


def manual_sample(parameters: tuple[float, ...], x0: float, x1: float,
                  target: float) -> tuple[float, float, tuple[float, ...]]:
    """手写相同网络的梯度，用作独立参照。

    :param parameters: 九个权重及偏置。
    :param x0: 第一个输入。
    :param x1: 第二个输入。
    :param target: 目标值。
    :return: 预测、损失和九个手写梯度。
    """
    p = parameters
    h1 = math.tanh(p[0] * x0 + p[1] * x1 + p[2])
    h2 = math.tanh(p[3] * x0 + p[4] * x1 + p[5])
    logit = p[6] * h1 + p[7] * h2 + p[8]
    prediction = 1 / (1 + math.exp(-logit))
    error = prediction - target
    output_delta = error * prediction * (1 - prediction)
    d1 = output_delta * p[6] * (1 - h1 * h1)
    d2 = output_delta * p[7] * (1 - h2 * h2)
    gradients = (d1 * x0, d1 * x1, d1, d2 * x0, d2 * x1, d2,
                 output_delta * h1, output_delta * h2, output_delta)
    return prediction, 0.5 * error * error, gradients


def mean_loss(parameters: tuple[float, ...]) -> float:
    """计算四条固定样本的平均损失。

    :param parameters: 当前九个参数。
    :return: 平均半平方误差。
    """
    return sum(xor_sample(parameters, *sample)[1] for sample in SAMPLES) / len(SAMPLES)


def train_epoch(parameters: tuple[float, ...], rate: float = 0.5) -> tuple[float, ...]:
    """执行一次全批量梯度下降。

    :param parameters: 更新前的九个参数。
    :param rate: 有限正学习率。
    :return: 更新后的九个参数。
    """
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError("学习率须为有限正数")
    gradient = [0.0] * len(parameters)
    for sample in SAMPLES:
        sample_gradient = xor_sample(parameters, *sample)[2]
        for i, value in enumerate(sample_gradient):
            gradient[i] += value / len(SAMPLES)
    updated = tuple(p - rate * g for p, g in zip(parameters, gradient))
    if any(not math.isfinite(p) for p in updated):
        raise ValueError("更新结果非有限")
    return updated


def main() -> None:
    """运行固定数据实验并按需输出 CSV。

    :return: 无。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", action="store_true",
                        help="输出跨语言比较用的字段和值")
    args = parser.parse_args()
    parameters = INITIAL
    initial = mean_loss(parameters)
    for _ in range(EPOCHS):
        parameters = train_epoch(parameters)
    final = mean_loss(parameters)
    if args.csv:
        print("field,value")
        print(f"loss_initial,{initial:.17g}")
        print(f"loss_final,{final:.17g}")
    else:
        print(f"XOR 初始平均损失 {initial:.8f}，{EPOCHS} 轮后 {final:.8f}")
    for index, sample in enumerate(SAMPLES):
        prediction = xor_sample(parameters, *sample)[0]
        if args.csv:
            print(f"prediction_{index},{prediction:.17g}")
        else:
            print(f"({sample[0]:.0f}, {sample[1]:.0f}) -> {prediction:.6f}，"
                  f"目标 {sample[2]:.0f}")


if __name__ == "__main__":
    main()
