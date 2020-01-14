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
from typing import Mapping, Tuple, TypeVar, Union, List

import numpy as np
import PIL.Image

from morphocut import Node, Output, RawOrVariable, ReturnOutputs, closing_if_closable
from morphocut._optional import import_optional_dependency

T = TypeVar("T")
MaybeTuple = Union[T, Tuple[T]]
MaybeList = Union[T, List[T]]


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
        fnames_images (Tuple, Variable, or a list thereof):
            Tuple of ``(filename, image)`` or a list of such tuples.
            ``filename`` is the name in the archive. ``image`` is a NumPy array.
            The file extension has to be one of ``".jpg"``, ``".png"`` or ``".gif"``
            to meet the specifications of EcoTaxa.
        meta (Mapping or Variable): Metadata to store in the TSV file.
        meta_fn (str, optional): TSV file. Must start with ``ecotaxa``.
        store_types (bool, optional): Whether to add a row with types after the header.
            Defaults to `True`, according to EcoTaxa's specifications.

    If multiple images are provided, ``image`` and
    ``image_name`` must be tuples of the same length.

    The TSV file will have the following columns by default:

    - ``img_file_name``: Name of the image file (including extension)
    - ``img_rank``: Rank of image to be displayed. Starts at 1.

    Other columns are read from ``meta``.

    Example:
        .. code-block:: python

            with Pipeline() as pipeline:
                image_fn = ...
                image = ImageReader(image_fn)
                meta = ... # Calculate some meta-data
                EcotaxaWriter("path/to/archive.zip", (image_fn, image), meta)
            pipeline.transform_stream()
    """

    def __init__(
        self,
        archive_fn: str,
        fnames_images: MaybeList[RawOrVariable[Tuple[str, ...]]],
        meta: RawOrVariable[Mapping],
        meta_fn: str = "ecotaxa_export.tsv",
        store_types: bool = True,
    ):
        super().__init__()
        self.archive_fn = archive_fn

        if isinstance(fnames_images, tuple):
            fnames_images = [fnames_images]

        if not isinstance(fnames_images, list):
            raise ValueError(
                "Unexpected type for fnames_images: needs to be a tuple or a list of tuples"
            )

        self.fnames_images = fnames_images
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
                fnames_images, meta = self.prepare_input(obj, ("fnames_images", "meta"))

                for img_rank, (fname, img) in enumerate(fnames_images, start=1):
                    img_ext = os.path.splitext(fname)[1]
                    pil_format = pil_extensions[img_ext]

                    img = PIL.Image.fromarray(img)
                    img_fp = io.BytesIO()
                    img.save(img_fp, format=pil_format)

                    zip_file.writestr(fname, img_fp.getvalue())

                    dataframe.append(
                        {**meta, "img_file_name": fname, "img_rank": img_rank}
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

    Example:
        .. code-block:: python

            with Pipeline() as p:
                image, meta = EcotaxaReader("path/to/archive.zip")
            p.transform_stream()
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
