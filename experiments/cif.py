from skimage.exposure import rescale_intensity
from skimage.io import imsave

from morphocut import LambdaNode, Node, Pipeline
from morphocut.io import ImageWriter
from morphocut.pims import BioformatsReader
from morphocut.str import Format
from morphocut.stream import TQDM, Slice

if __name__ == "__main__":
    with Pipeline() as p:
        input_fn = "/home/moi/Work/Datasets/06_CD20_brightfield_6.cif"

        frame, series = BioformatsReader(input_fn, meta=False)

        # Every second frame is in fact a mask
        # TODO: Batch consecutive objects in the stream

        Slice(None, None, 2)

        output_fn = Format("/tmp/cif/{}.png", series)

        frame = LambdaNode(rescale_intensity, frame)

        LambdaNode(imsave, output_fn, frame)

        TQDM()

    print(p)

    p.run()
