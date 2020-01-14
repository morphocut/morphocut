import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from morphocut import Node, Output, RawOrVariable, ReturnOutputs, Variable


@ReturnOutputs
@Output("value")
class Plot(Node):
    def __init__(self, x, vline=None, **kwargs):
        super().__init__()
        self.x = x
        self.vline = vline
        self.kwargs = kwargs

    def transform(self, x, vline):
        fig = Figure(figsize=(5, 4), dpi=100)
        canvas = FigureCanvasAgg(fig)

        # plot
        ax = fig.gca()
        ax.plot(x, **self.kwargs)

        if vline is not None:
            ax.axvline(vline)

        fig.tight_layout()
        canvas.draw()
        s, (width, height) = canvas.print_to_buffer()
        return np.fromstring(s, np.uint8).reshape((height, width, 4))


@ReturnOutputs
@Output("value")
class Bar(Node):
    def __init__(self, x, height, vline=None, **kwargs):
        super().__init__()
        self.x = x
        self.height = height
        self.vline = vline
        self.kwargs = kwargs

    def transform(self, x, height, vline):
        fig = Figure(figsize=(5, 4), dpi=100)
        canvas = FigureCanvasAgg(fig)

        # plot
        ax = fig.gca()
        ax.bar(x, height, **self.kwargs)

        if vline is not None:
            ax.axvline(vline, c="r")

        fig.tight_layout()
        canvas.draw()
        s, (width, height) = canvas.print_to_buffer()
        return np.fromstring(s, np.uint8).reshape((height, width, 4))
