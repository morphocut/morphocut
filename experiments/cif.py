from itertools import repeat, islice

import numpy as np

from morphocut import Call, Node, Pipeline, ReturnOutputs, Variable
from morphocut.image import ImageWriter, RescaleIntensity
from morphocut.pims import BioformatsReader
from morphocut.str import Format
from morphocut.stream import TQDM, Slice


@ReturnOutputs
class Pack(Node):
    """
    Pack values of subsequent objects of the stream into one Iterable.

    See Also:
        :py:class:`morphocut.stream.Unpack`
    """

    def __init__(self, size, *variables):
        super().__init__()
        self.size = size
        self.variables = variables
        # Mess with self.outputs
        self.outputs = [Variable(v.name, self) for v in self.variables]

    def transform_stream(self, stream):
        while True:
            packed = list(islice(stream, self.size))

            if not packed:
                break

            packed_values = tuple(tuple(o[v] for o in packed) for v in self.variables)

            yield self.prepare_output(packed[0], *packed_values)


if __name__ == "__main__":
    with Pipeline() as p:
        input_fn = "/home/moi/Work/0-Datasets/06_CD20_brightfield_6.cif"

        frame, series = BioformatsReader(input_fn, meta=False)

        # Every second frame is in fact a mask
        # TODO: Batch consecutive objects in the stream

        frame, mask = Pack(2, frame).unpack(2)

        image_fn = Format("/tmp/cif/{}-img.png", series)
        mask_fn = Format("/tmp/cif/{}-mask.png", series)

        frame = RescaleIntensity(frame, dtype=np.uint8)
        mask = RescaleIntensity(mask, dtype=np.uint8)

        ImageWriter(image_fn, frame)
        ImageWriter(mask_fn, mask)

        TQDM()

    print(p)

    p.run()
