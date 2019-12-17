from morphocut import Node, Output, RawOrVariable, ReturnOutputs


@ReturnOutputs
@Output("out")
class ExponentialSmoothing(Node):
    """
    Smooth values with exponential decay.

    Formula: alpha * value + (1 - alpha) * last_value

    Args:
        value (Variable): Current value.
        alpha (float): Alpha is the Smoothing factor, larger values of alpha actually reduce the level of 
            smoothing. It should be (0 < alpha < 1)
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
