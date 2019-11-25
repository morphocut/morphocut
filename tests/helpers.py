from morphocut import Node, Output, ReturnOutputs
from skimage.data import binary_blobs
import numpy as np


@ReturnOutputs
@Output("blobs")
class BinaryBlobs(Node):
    def __init__(
        self, length=512, blob_size_fraction=0.1, n_dim=2, volume_fraction=0.5
    ):
        super().__init__()

        self.length = length
        self.blob_size_fraction = blob_size_fraction
        self.n_dim = n_dim
        self.volume_fraction = volume_fraction

    def transform(self):
        return binary_blobs(
            self.length, self.blob_size_fraction, self.n_dim, self.volume_fraction
        )


@ReturnOutputs
@Output("blobs")
class NoiseImage(Node):
    def __init__(self, shape):
        super().__init__()

        self.shape = shape

    def transform(self, shape):
        return np.random.rand(*shape)


@ReturnOutputs
@Output("value")
class Const(Node):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def transform(self, value):
        return value
