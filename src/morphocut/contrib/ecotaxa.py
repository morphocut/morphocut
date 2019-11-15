"""
Read and write EcoTaxa archives.

    "`EcoTaxa`_ is a web application dedicated to the visual exploration
    and the taxonomic annotation of images that illustrate the
    beauty of planktonic biodiversity."

.. _EcoTaxa: https://ecotaxa.obs-vlfr.fr/
"""
import fnmatch
import io
import os.path
import zipfile
from typing import Mapping, Tuple, TypeVar, Union

import numpy as np
import PIL.Image

from morphocut import Node, Output, RawOrVariable, ReturnOutputs, closing_if_closable
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
    """
    Create an archive of images and metadata that is importable to EcoTaxa.

    Args:
        archive_fn (str): Location of the output file.
        image (np.array, Variable, or a tuple thereof): One or more images.
        image_name (str, Variable, or a tuple thereof): One or more image names.
        meta (Mapping or Variable): Metadata to store in the TSV file.
        image_ext (str, optional): Image extension, will be appended to ``image_name``.
            Has to be one of ``".jpg"``, ``".png"`` or ``".gif"``.
        meta_fn (str, optional): TSV file. Must start with ``ecotaxa``.
        store_types (bool, optional): Whether to add a row with types after the header.

    If multiple images are provided, ``image`` and
    ``image_name`` must be tuples of the same length.

    The TSV file will have the following columns by default:

    - ``img_file_name``: Name of the image file (including extension)
    - ``img_rank``: Rank of image to be displayed. Starts at 1.

    Other columns are read from ``meta``.
    """

    def __init__(
        self,
        archive_fn: str,
        image: MaybeTuple[RawOrVariable],
        image_name: MaybeTuple[RawOrVariable[str]],
        meta: RawOrVariable[Mapping],
        image_ext: MaybeTuple[str] = ".jpg",
        meta_fn: str = "ecotaxa_export.tsv",
        store_types: bool = True,
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
        self.store_types = store_types

        self._pd = import_optional_dependency("pandas")

    def transform_stream(self, stream):
        pil_extensions = PIL.Image.registered_extensions()

        with closing_if_closable(stream), zipfile.ZipFile(
            self.archive_fn, mode="w"
        ) as zip_file:
            dataframe = []
            i = 0
            for obj in stream:
                image, image_name, meta = self.prepare_input(
                    obj, ("image", "image_name", "meta")
                )

                for img_rank, (img, img_name, img_ext) in enumerate(
                    zip(image, image_name, self.image_ext), start=1
                ):
                    pil_format = pil_extensions[img_ext]

                    img = PIL.Image.fromarray(img)
                    img_fp = io.BytesIO()
                    img.save(img_fp, format=pil_format)

                    arcname = img_name + img_ext

                    zip_file.writestr(arcname, img_fp.getvalue())

                    dataframe.append(
                        {**meta, "img_file_name": arcname, "img_rank": img_rank}
                    )

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
@Output("image")
@Output("meta")
class EcotaxaReader(Node):
    """
    |stream| Read an archive of images and metadata that is importable to EcoTaxa.

    Args:
        archive_fn (str, Variable): Location of the archive file.
        img_rank (int, Variable, or a tuple thereof, optional): One or more image ranks.

    Returns:
        (image, meta): A tuple of image(s) and metadata.

    To read multiple image ranks, provide a tuple of ints as ``img_rank``.
    The first output will then be a tuple of images.

    The TSV file needs at least an ``img_file_name``
    column that provides the name of the image file.
    Other columns are read from ``meta``.

    The TSV file MAY contain a row of types after the header
    (``"[f]"`` for numeric columns, ``"[t]"`` else).
    """

    def __init__(
        self,
        archive_fn: RawOrVariable[str],
        img_rank: MaybeTuple[RawOrVariable[int]] = 1,
    ):
        super().__init__()
        self.archive_fn = archive_fn
        self.img_rank = img_rank
        self._pd = import_optional_dependency("pandas")

    def transform_stream(self, stream):
        with closing_if_closable(stream):
            for obj in stream:
                archive_fn, img_rank = self.prepare_input(
                    obj, ("archive_fn", "img_rank")
                )

                with zipfile.ZipFile(archive_fn, mode="r") as zip_file:
                    index_names = fnmatch.filter(zip_file.namelist(), "ecotaxa_*")

                    for index_name in index_names:
                        index_base = os.path.dirname(index_name)
                        with zip_file.open(index_name) as index_fp:
                            dataframe = self._pd.read_csv(index_fp, sep="\t")
                            dataframe = self._fix_types(dataframe)

                            for _, row in dataframe.iterrows():
                                image_fn = os.path.join(
                                    index_base, row["img_file_name"]
                                )

                                with zip_file.open(image_fn) as image_fp:
                                    image = np.array(PIL.Image.open(image_fp))

                                yield self.prepare_output(
                                    obj.copy(), image, row.to_dict()
                                )

    def _fix_types(self, dataframe):
        first_row = dataframe.iloc[0]

        num_cols = []
        for c, v in first_row.items():
            if v == "[f]":
                num_cols.append(c)
            elif v == "[t]":
                continue
            else:
                # If the first row contains other values than [f] or [t],
                # it is not a type header and the dataframe doesn't need to be changed.
                return dataframe

        dataframe = dataframe.iloc[1:].copy()

        dataframe[num_cols] = dataframe[num_cols].apply(
            self._pd.to_numeric, errors="coerce", axis=1
        )

        return dataframe
