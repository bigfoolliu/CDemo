"""固定容量张量图：从 M03 标量节点推进到 M02 批量层。"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import math

MAX_ELEMENTS = 6
GRAPH_CAPACITY = 32
INPUTS = (1.0, 2.0, -1.0, 1.0)
WEIGHTS = (1.0, -1.0, 2.0, 0.5)
BIAS = (0.5, -0.5)


@dataclass
class Node:
    """保存张量节点的前向值、梯度、形状与父索引。"""

    rows: int
    cols: int
    values: list[float]
    op: str = "leaf"
    left: int = -1
    right: int = -1
    broadcast_right: bool = False
    gradients: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        """按张量大小初始化梯度数组。

        :return: 无。
        """
        self.gradients = [0.0] * (self.rows * self.cols)


class Graph:
    """固定节点容量的前向图；索引只在本次图生命周期内有效。"""

    def __init__(self, capacity: int = GRAPH_CAPACITY) -> None:
        """创建空图。

        :param capacity: 最大节点数，范围为 1 至 GRAPH_CAPACITY。
        :return: 无。
        :raises ValueError: 容量越界时抛出。
        """
        if not 1 <= capacity <= GRAPH_CAPACITY:
            raise ValueError("图容量无效")
        self.capacity = capacity
        self.nodes: list[Node] = []

    def release(self) -> None:
        """释放图持有的节点；旧索引随后失效。

        :return: 无。
        """
        self.nodes.clear()

    def _push(self, node: Node) -> int:
        """在全部检查成功后加入节点。

        :param node: 待加入的节点。
        :return: 新节点索引。
        :raises ValueError: 形状、数值或容量无效时抛出。
        """
        if (len(self.nodes) >= self.capacity or node.rows < 1 or node.cols < 1 or
                node.rows * node.cols > MAX_ELEMENTS or
                len(node.values) != node.rows * node.cols or
                any(not math.isfinite(v) for v in node.values)):
            raise ValueError("节点容量、形状或数值无效")
        self.nodes.append(node)
        return len(self.nodes) - 1

    def leaf(self, rows: int, cols: int, values: tuple[float, ...]) -> int:
        """复制有限数据建立叶张量。

        :param rows: 行数，向量用一行。
        :param cols: 列数，标量为 1×1。
        :param values: 行优先的数值。
        :return: 新节点索引。
        :raises ValueError: 形状、数值或容量无效时抛出。
        """
        return self._push(Node(rows, cols, list(values)))

    def binary(self, op: str, left: int, right: int) -> int:
        """创建加法、逐元素乘法或矩阵乘法节点。

        :param op: add、mul 或 matmul；add 支持右侧行向量广播。
        :param left: 左父节点索引。
        :param right: 右父节点索引，可与左侧相同。
        :return: 新节点索引。
        :raises ValueError: 操作、索引、形状或结果无效时抛出。
        """
        if op not in ("add", "mul", "matmul") or not 0 <= left < len(self.nodes) or not 0 <= right < len(self.nodes):
            raise ValueError("操作或父索引无效")
        a, b = self.nodes[left], self.nodes[right]
        broadcast = op == "add" and a.rows > 1 and b.rows == 1 and a.cols == b.cols
        if op == "matmul":
            if a.cols != b.rows:
                raise ValueError("矩阵乘法内维不等")
            rows, cols = a.rows, b.cols
        else:
            if (a.rows, a.cols) != (b.rows, b.cols) and not broadcast:
                raise ValueError("逐元素形状不匹配")
            rows, cols = a.rows, a.cols
        values = [0.0] * (rows * cols)
        for i in range(rows):
            for j in range(cols):
                index = i * cols + j
                if op == "matmul":
                    for k in range(a.cols):
                        values[index] += a.values[i * a.cols + k] * b.values[k * b.cols + j]
                else:
                    other = j if broadcast else index
                    values[index] = (a.values[index] + b.values[other] if op == "add"
                                     else a.values[index] * b.values[other])
        return self._push(Node(rows, cols, values, op, left, right, broadcast))

    def unary(self, op: str, source: int) -> int:
        """创建求和、均值或 ReLU 节点；零点导数定为零。

        :param op: sum、mean 或 relu。
        :param source: 输入节点索引。
        :return: 新节点索引。
        :raises ValueError: 操作、索引或结果无效时抛出。
        """
        if op not in ("sum", "mean", "relu") or not 0 <= source < len(self.nodes):
            raise ValueError("操作或输入索引无效")
        a = self.nodes[source]
        if op == "relu":
            return self._push(Node(a.rows, a.cols, [max(0.0, v) for v in a.values], op, source))
        total = sum(a.values)
        value = total / len(a.values) if op == "mean" else total
        return self._push(Node(1, 1, [value], op, source))

    def backward(self, output: int) -> None:
        """清零后从标量输出逆序累加梯度。

        :param output: 有效 1×1 输出索引。
        :return: 无；相同图重复调用给出相同梯度。
        :raises ValueError: 输出或梯度无效时抛出。
        """
        if not 0 <= output < len(self.nodes) or (self.nodes[output].rows, self.nodes[output].cols) != (1, 1):
            raise ValueError("反向输出必须为现存标量")
        for node in self.nodes:
            node.gradients = [0.0] * len(node.values)
        self.nodes[output].gradients[0] = 1.0
        for index in range(output, -1, -1):
            node = self.nodes[index]
            if node.op == "leaf":
                continue
            a = self.nodes[node.left]
            b = self.nodes[node.right] if node.right >= 0 else None
            for i in range(node.rows):
                for j in range(node.cols):
                    at = i * node.cols + j
                    upstream = node.gradients[at]
                    if node.op in ("add", "mul"):
                        assert b is not None
                        other = j if node.broadcast_right else at
                        a.gradients[at] += upstream * (b.values[other] if node.op == "mul" else 1)
                        b.gradients[other] += upstream * (a.values[at] if node.op == "mul" else 1)
                    elif node.op == "matmul":
                        assert b is not None
                        for k in range(a.cols):
                            a.gradients[i * a.cols + k] += upstream * b.values[k * b.cols + j]
                            b.gradients[k * b.cols + j] += upstream * a.values[i * a.cols + k]
            if node.op in ("sum", "mean", "relu"):
                for i, value in enumerate(a.values):
                    if node.op == "relu":
                        a.gradients[i] += node.gradients[i] * (1 if value > 0 else 0)
                    else:
                        a.gradients[i] += node.gradients[0] / (len(a.values) if node.op == "mean" else 1)
        if any(not math.isfinite(v) for node in self.nodes for v in node.gradients):
            for node in self.nodes:
                node.gradients = [0.0] * len(node.values)
            raise ValueError("反向梯度非有限")


def dense_run(inputs: tuple[float, ...] = INPUTS,
              weights: tuple[float, ...] = WEIGHTS,
              bias: tuple[float, ...] = BIAS) -> dict[str, float]:
    """用图复现 M02 两样本的前向、反向和学习率 0.1 更新。

    :param inputs: 行优先 2×2 输入。
    :param weights: 行优先 2×2 权重。
    :param bias: 长度 2 的偏置行向量。
    :return: 损失、预测、梯度和更新后的参数字段。
    :raises ValueError: 形状、数值或计算失败时抛出。
    """
    graph = Graph()
    try:
        x = graph.leaf(2, 2, inputs)
        w = graph.leaf(2, 2, weights)
        b = graph.leaf(1, 2, bias)
        target = graph.leaf(2, 2, (-2.0, 0.0, 0.0, -1.0))
        half = graph.leaf(1, 1, (0.5,))
        product = graph.binary("matmul", x, w)
        prediction = graph.binary("add", product, b)
        error = graph.binary("add", prediction, target)
        squared = graph.binary("mul", error, error)
        mean = graph.unary("mean", squared)
        loss = graph.binary("mul", mean, half)
        graph.backward(loss)
        result = {"loss": graph.nodes[loss].values[0]}
        for i in range(4):
            result[f"prediction_{i}"] = graph.nodes[prediction].values[i]
            result[f"input_grad_{i}"] = graph.nodes[x].gradients[i]
            result[f"weight_grad_{i}"] = graph.nodes[w].gradients[i]
            result[f"updated_weight_{i}"] = weights[i] - 0.1 * result[f"weight_grad_{i}"]
        for i in range(2):
            result[f"bias_grad_{i}"] = graph.nodes[b].gradients[i]
            result[f"updated_bias_{i}"] = bias[i] - 0.1 * result[f"bias_grad_{i}"]
        return result
    finally:
        graph.release()


def main() -> None:
    """输出固定实验或跨语言 CSV。

    :return: 无。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", action="store_true", help="输出 field,value")
    args = parser.parse_args()
    values = dense_run()
    if args.csv:
        print("field,value")
        for field, value in values.items():
            print(f"{field},{value:.17g}")
    else:
        print("张量图复现 M02：损失={:.5f}，预测={}".format(
            values["loss"], [values[f"prediction_{i}"] for i in range(4)]))
        print("权重梯度={}；偏置梯度={}".format(
            [values[f"weight_grad_{i}"] for i in range(4)],
            [values[f"bias_grad_{i}"] for i in range(2)]))
        print("学习率 0.1 更新后：W={}，b={}".format(
            [values[f"updated_weight_{i}"] for i in range(4)],
            [values[f"updated_bias_{i}"] for i in range(2)]))


if __name__ == "__main__":
    main()
