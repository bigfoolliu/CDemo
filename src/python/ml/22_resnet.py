"""对比普通映射与残差映射的前向值和反向梯度路径。"""


def main() -> None:
    """计算一个标量残差块并用链式法则展示恒等捷径。

    :return: 无。
    """
    input_value = 0.7
    weight = 0.2
    residual = weight * input_value
    output = input_value + residual
    loss_gradient = 1.0
    gradient_input = loss_gradient * (1.0 + weight)
    gradient_weight = loss_gradient * input_value
    print("残差块 y=x+F(x)：x={:.2f}, F(x)={:.2f}, y={:.2f}".format(
        input_value, residual, output))
    print("若 L=y，则 dL/dx=1+dF/dx={:.2f}，其中 1 来自恒等捷径。".format(gradient_input))
    print("dL/dw={:.2f}；实际网络使用张量卷积和非线性层。".format(gradient_weight))


if __name__ == "__main__":
    main()
