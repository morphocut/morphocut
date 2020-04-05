from morphocut import Node, Output, RawOrVariable, ReturnOutputs


@ReturnOutputs
@Output("out")
class ExponentialSmoothing(Node):
    """
    Smooth values with exponential decay.

    Formula: alpha * value + (1 - alpha) * last_value

    Args:
        value (Variable): Current value.
        alpha (float): Rate of adjustment (0 <= alpha <= 1). Smaller values lead to more smoothing.
    """

    def __init__(self, value: RawOrVariable, alpha: float):
        super().__init__()

        self.value = value
        self.alpha = alpha
        self.last_value = None

    def transform(self, value):
        if self.last_value is None:
            self.last_value = value
        else:
            self.last_value = self.alpha * value + (1 - self.alpha) * self.last_value

        return self.last_value
