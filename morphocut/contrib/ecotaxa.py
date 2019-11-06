"""
Read and write EcoTaxa archives.

    "`EcoTaxa`_ is a web application dedicated to the visual exploration
    and the taxonomic annotation of images that illustrate the
    beauty of planktonic biodiversity."

.. _EcoTaxa: https://ecotaxa.obs-vlfr.fr/
"""
import io
import zipfile
from typing import Mapping, Tuple, TypeVar, Union

import numpy as np
import PIL

from morphocut import Node, RawOrVariable, ReturnOutputs
from morphocut._optional import import_optional_dependency

T = TypeVar("T")
MaybeTuple = Union[T, Tuple[T]]


def dtype_to_ecotaxa(dtype):
    try:
        if np.issubdtype(dtype, np.number):
            return "[f]"
    except TypeError:
        print(type(dtype))
        raise

    return "[t]"


@ReturnOutputs
class EcotaxaWriter(Node):
    """Zip the image and its meta data."""

    def __init__(
        self,
        archive_fn: str,
        image: MaybeTuple[RawOrVariable],
        image_name: MaybeTuple[RawOrVariable[str]],
        meta: RawOrVariable[Mapping],
        image_ext: MaybeTuple[str] = ".jpg",
        meta_fn: str = "ecotaxa_export.tsv",
    ):
        super().__init__()
        self.archive_fn = archive_fn

        if not isinstance(image, tuple):
            image = (image,)

        if not isinstance(image_name, tuple):
            image_name = (image_name,)

        if len(image) != len(image_name):
            raise ValueError("Length of `image` and `image_name` do not match")

        if not isinstance(image_ext, tuple):
            image_ext = (image_ext,) * len(image)

        self.image = image
        self.image_name = image_name
        self.image_ext = image_ext

        self.meta = meta

        self.meta_fn = meta_fn

        self._pd = import_optional_dependency("pandas")

    def transform_stream(self, stream):
        pil_extensions = PIL.Image.registered_extensions()

        with zipfile.ZipFile(self.archive_fn, mode="w") as zip_file:
            dataframe = []
            i = 0
            for obj in stream:
                image, image_name, meta = self.prepare_input(
                    obj, ("image", "image_name", "meta")
                )

                for img, img_name, img_ext in zip(image, image_name, self.image_ext):
                    pil_format = pil_extensions[img_ext]

                    img = PIL.Image.fromarray(img)
                    img_fp = io.BytesIO()
                    img.save(img_fp, format=pil_format)

                    arcname = img_name + img_ext

                    zip_file.writestr(arcname, img_fp.getvalue())

                    dataframe.append({**meta, "img_file_name": arcname})

                yield obj

                i += 1

            dataframe = self._pd.DataFrame(dataframe)

            # Insert types into header
            type_header = [dtype_to_ecotaxa(dt) for dt in dataframe.dtypes]
            dataframe.columns = self._pd.MultiIndex.from_tuples(
                list(zip(dataframe.columns, type_header))
            )

            zip_file.writestr(
                self.meta_fn, dataframe.to_csv(sep="\t", encoding="utf-8", index=False)
            )

            print("Wrote {:,d} objects to {}.".format(i, self.archive_fn))


@ReturnOutputs
class EcotaxaReader(Node):
    ...
