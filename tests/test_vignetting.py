from skimage.io import imread

from morphocut import LambdaNode, Pipeline
from morphocut.stream import FromIterable
from morphocut.vignetting import VignettingCorrector


def test_vignette_corrector_no_channel(image_fns):

    with Pipeline() as pipeline:
        img_fn = FromIterable(image_fns)
        image = LambdaNode(imread, img_fn, as_gray=True)
        result = VignettingCorrector(image)

    stream = pipeline.transform_stream()
    obj = next(stream)
