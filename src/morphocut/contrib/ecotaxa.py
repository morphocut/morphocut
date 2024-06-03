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
import pathlib
import tarfile
import zipfile
from abc import ABC, abstractproperty
from shutil import copyfileobj
from typing import (
    IO,
    BinaryIO,
    Callable,
    List,
    Mapping,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

import numpy as np
import pandas as pd
import PIL
import PIL.Image

from morphocut import Node, Output, RawOrVariable, ReturnOutputs, closing_if_closable
from morphocut.core import Stream
from morphocut.utils import StreamEstimator, stream_groupby

T = TypeVar("T")
MaybeTuple = Union[T, Tuple[T]]
MaybeList = Union[T, List[T]]


def dtype_to_ecotaxa(dtype):
    try:
        if np.issubdtype(dtype, np.number):
            return "[f]"
    except TypeError:  # pragma: no cover
        print(type(dtype))
        raise

    return "[t]"


class MemberNotFoundError(Exception):
    pass


class UnknownArchiveError(Exception):
    pass


class Archive:
    """
    A generic archive reader and writer for ZIP and TAR archives.
    """

    extensions: List[str] = []

    def __new__(cls, archive_fn: Union[str, pathlib.Path], mode: str = "r"):
        archive_fn = str(archive_fn)

        if mode[0] == "r":
            for subclass in cls.__subclasses__():
                if subclass.is_readable(archive_fn):
                    return super(Archive, subclass).__new__(subclass)

            raise UnknownArchiveError(f"No handler found to read {archive_fn}")

        if mode[0] in ("a", "w", "x"):
            for subclass in cls.__subclasses__():
                if any(archive_fn.endswith(ext) for ext in subclass.extensions):
                    return super(Archive, subclass).__new__(subclass)

            raise UnknownArchiveError(f"No handler found to write {archive_fn}")

    @staticmethod
    def is_readable(archive_fn) -> bool:
        raise NotImplementedError()

    def __init__(self, archive_fn: Union[str, pathlib.Path], mode: str = "r"):
        raise NotImplementedError()

    def read_member(self, member_fn) -> IO:
        """
        Raises:
            MemberNotFoundError: if a member was not found
        """
        raise NotImplementedError()

    def write_member(
        self, member_fn, fileobj_or_bytes: Union[IO, bytes], compress_hint=True
    ):
        raise NotImplementedError()

    def find(self, pattern) -> List[str]:
        return fnmatch.filter(self.members(), pattern)

    def members(self) -> List[str]:
        raise NotImplementedError()

    def close(self):  # pragma: no cover
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_, **__):
        self.close()


class TarArchive(Archive):
    extensions = [
        ".tar",
        ".tar.bz2",
        ".tb2",
        ".tbz",
        ".tbz2",
        ".tz2",
        ".tar.gz",
        ".taz",
        ".tgz",
        ".tar.lzma",
        ".tlz",
    ]

    @staticmethod
    def is_readable(archive_fn):
        return tarfile.is_tarfile(archive_fn)

    def __init__(self, archive_fn: Union[str, pathlib.Path], mode: str = "r"):
        self._tar = tarfile.open(archive_fn, mode)
        self._members = None

    def close(self):
        self._tar.close()

    def read_member(self, member_fn):
        return self._tar.extractfile(self._resolve_member(member_fn))

    def _load_members(self):
        if self._members is not None:
            return

        self._members = {tar_info.name: tar_info for tar_info in self._tar.getmembers()}

    def _resolve_member(self, member):
        if isinstance(member, tarfile.TarInfo):
            return member

        self._load_members()
        assert self._members is not None

        return self._members[member]

    def write_member(
        self, member_fn: str, fileobj_or_bytes: Union[IO, bytes], compress_hint=True
    ):
        if isinstance(fileobj_or_bytes, bytes):
            fileobj_or_bytes = io.BytesIO(fileobj_or_bytes)

        if isinstance(fileobj_or_bytes, io.BytesIO):
            tar_info = tarfile.TarInfo(member_fn)
            tar_info.size = len(fileobj_or_bytes.getbuffer())
        else:
            tar_info = self._tar.gettarinfo(arcname=member_fn, fileobj=fileobj_or_bytes)

        self._tar.addfile(tar_info, fileobj=fileobj_or_bytes)

    def members(self):
        return self._tar.getnames()


class ZipArchive(Archive):
    extensions = [".zip"]

    @staticmethod
    def is_readable(archive_fn):
        return zipfile.is_zipfile(archive_fn)

    def __init__(self, archive_fn: Union[str, pathlib.Path], mode: str = "r"):
        self._zip = zipfile.ZipFile(archive_fn, mode)

    def members(self):
        return self._zip.namelist()

    def read_member(self, member):
        try:
            return self._zip.open(member)
        except KeyError as exc:
            raise MemberNotFoundError(f"{member} not in {self._zip.filename}") from exc

    def write_member(
        self, member_fn: str, fileobj_or_bytes: Union[IO, bytes], compress_hint=True
    ):
        compress_type = zipfile.ZIP_DEFLATED if compress_hint else zipfile.ZIP_STORED
        # TODO: Optimize for on-disk files and BytesIO (.getvalue())
        if isinstance(fileobj_or_bytes, bytes):
            return self._zip.writestr(
                member_fn, fileobj_or_bytes, compress_type=compress_type
            )

        self._zip.writestr(
            member_fn, fileobj_or_bytes.read(), compress_type=compress_type
        )

    def close(self):
        self._zip.close()


def split_path(path: str) -> Tuple[str, str]:
    if "/" in path:
        head, tail = path.rsplit("/", 1)
        return head, tail
    return "", path


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
        meta (Mapping or Variable, optional): Metadata to store in the TSV file.
            Each key corresponds to a column in the resulting file.
        object_meta (Mapping or Variable, optional): Metadata stored with ``object_`` prefix.
        acq_meta (Mapping or Variable, optional): Metadata stored with ``acq_`` prefix.
        process_meta (Mapping or Variable, optional): Metadata stored with ``process_`` prefix.
        sample_meta (Mapping or Variable, optional): Metadata stored with ``sample_`` prefix.
        meta_fn (str, optional): TSV file. Must start with ``ecotaxa``.
        store_types (bool, optional): Whether to add a row with types after the header.
            Defaults to `True`, according to EcoTaxa's specifications.

    If multiple images are provided, ``image`` and
    ``image_name`` must be tuples of the same length.

    The TSV file will have the following columns by default:

    - ``img_file_name``: Name of the image file (including extension)
    - ``img_rank``: Rank of image to be displayed. Starts at 1.

    Other columns are read from ``meta``. The file will contain a column for each object in the stream.

    Example:
        .. code-block:: python

            with Pipeline() as pipeline:
                image_fn = ...
                image = ImageReader(image_fn)
                meta = ... # Calculate some meta-data
                EcotaxaWriter("path/to/archive.zip", (image_fn, image), meta)
            pipeline.transform_stream()

    .. seealso::
        For more information about the metadata fields, see the project import page of EcoTaxa.
    """

    def __init__(
        self,
        archive_fn: RawOrVariable[str],
        fnames_images: MaybeList[RawOrVariable[Tuple[str, ...]]],
        meta: Optional[RawOrVariable[Mapping]] = None,
        object_meta: Optional[RawOrVariable[Mapping]] = None,
        acq_meta: Optional[RawOrVariable[Mapping]] = None,
        process_meta: Optional[RawOrVariable[Mapping]] = None,
        sample_meta: Optional[RawOrVariable[Mapping]] = None,
        meta_fn: RawOrVariable[str] = "ecotaxa_export.tsv",
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
        self.object_meta = object_meta
        self.acq_meta = acq_meta
        self.process_meta = process_meta
        self.sample_meta = sample_meta
        self.meta_fn = meta_fn
        self.store_types = store_types

    @classmethod
    def _prepare_images(
        cls, fnames_images, archive: Archive, pil_extensions, meta_prefix, meta
    ):
        for img_rank, (fname, img) in enumerate(fnames_images, start=1):
            if isinstance(img, io.IOBase):
                # Stream
                img_fp = img

            else:
                # Array
                img = np.asarray(img)

                img_ext = os.path.splitext(fname)[1]
                pil_format = pil_extensions[img_ext]

                img = PIL.Image.fromarray(img)
                img_fp = io.BytesIO()
                try:
                    img.save(img_fp, format=pil_format)
                except:  # pragma: no cover
                    print(f"EcotaxaWriter: Error writing {fname}")
                    raise

            # Rewind
            img_fp.seek(0)
            # Do not compress image files as already compressed
            archive.write_member(meta_prefix + fname, img_fp, compress_hint=False)

            yield {
                **meta,
                "img_file_name": fname,
                "img_rank": img_rank,
            }

    def transform_stream(self, stream):
        PIL.Image.init()
        pil_extensions = PIL.Image.registered_extensions()

        with closing_if_closable(stream):
            for archive_fn, archive_group in stream_groupby(stream, by=self.archive_fn):
                i = 0
                with Archive(archive_fn, "w") as archive:
                    for meta_fn, meta_group in stream_groupby(
                        archive_group, by=self.meta_fn
                    ):
                        meta_fn: str
                        dataframe = []
                        meta_prefix = split_path(meta_fn)[0]
                        if meta_prefix:
                            meta_prefix = meta_prefix + "/"

                        for obj in meta_group:
                            (
                                fnames_images,
                                meta,
                                object_meta,
                                acq_meta,
                                process_meta,
                                sample_meta,
                            ) = self.prepare_input(
                                obj,
                                (
                                    "fnames_images",
                                    "meta",
                                    "object_meta",
                                    "acq_meta",
                                    "process_meta",
                                    "sample_meta",
                                ),
                            )  # type: ignore

                            if meta is None:
                                meta = {}

                            if object_meta is not None:
                                meta.update(
                                    ("object_" + k, v) for k, v in object_meta.items()
                                )

                            if acq_meta is not None:
                                meta.update(
                                    ("acq_" + k, v) for k, v in acq_meta.items()
                                )

                            if process_meta is not None:
                                meta.update(
                                    ("process_" + k, v) for k, v in process_meta.items()
                                )

                            if sample_meta is not None:
                                meta.update(
                                    ("sample_" + k, v) for k, v in sample_meta.items()
                                )

                            if fnames_images:
                                # Metadata and images: Store image and repeat meta for each individual image
                                dataframe.extend(
                                    self._prepare_images(
                                        fnames_images,
                                        archive,
                                        pil_extensions,
                                        meta_prefix,
                                        meta,
                                    )
                                )
                            else:
                                # Only metadata: Write only meta
                                dataframe.append(meta)

                            yield obj

                            i += 1

                        dataframe = pd.DataFrame(dataframe)

                        if dataframe.size:
                            # Insert types into header
                            type_header = [
                                dtype_to_ecotaxa(dt) for dt in dataframe.dtypes
                            ]
                            dataframe.columns = pd.MultiIndex.from_tuples(
                                list(zip(dataframe.columns, type_header))
                            )

                            archive.write_member(
                                meta_fn,
                                io.BytesIO(
                                    dataframe.to_csv(
                                        sep="\t", encoding="utf-8", index=False
                                    ).encode()
                                ),
                            )

                        print(
                            f"EcotaxaWriter: Wrote {len(dataframe):,d} entries to {meta_fn}."
                        )

                print(f"EcotaxaWriter: Wrote {i:,d} objects to {archive_fn}.")


class EcotaxaObject:
    """
    Attributes:
        meta (Mapping): Object metadata
        index_fn (str, optional): Source index filename inside the archive.
        archive_fn (str, optional): Sourche archive filename.
        default_mode (str, optional): Default mode for image loading.
                See `PIL Modes <https://pillow.readthedocs.io/en/stable/handbook/concepts.html#modes>`_.

    """

    __slots__ = ["meta", "_image_data", "index_fn", "archive_fn", "default_mode"]

    def __init__(
        self, meta, image_data, archive_fn=None, index_fn=None, default_mode=None
    ):
        self.meta = meta
        self._image_data = image_data
        self.archive_fn = archive_fn
        self.index_fn = index_fn
        self.default_mode = default_mode

    @property
    def image(self):
        return self.get_image()

    @property
    def image_data(self):
        return self.get_image_data()

    @property
    def object_id(self):
        return self.meta.get("object_id", "<unidentified object>")

    def get_image_data(self, img_rank=None):
        if img_rank is None:
            return next(iter(self._image_data.values()))
        else:
            try:
                return self._image_data[img_rank]
            except KeyError:
                print(
                    f"Unknown rank {img_rank}. Available:",
                    list(self._image_data.keys()),
                )
                raise

    def get_image(self, img_rank=None, mode=None):
        """
        Get an image for the object.

        Args:
            img_rank: Rank of the image. (Default: The first one.)
            mode: Mode to load the image.
                See `PIL Modes <https://pillow.readthedocs.io/en/stable/handbook/concepts.html#modes>`_.
        """

        data = self.get_image_data(img_rank)

        if mode is None:
            mode = self.default_mode

        try:
            image = PIL.Image.open(data)
            if mode is not None:
                image = image.convert(mode)
            return np.array(image)
        except PIL.UnidentifiedImageError:
            head = data.getvalue()[:20]
            print(f"Error reading {self.object_id}: {head}")
            raise


@ReturnOutputs
@Output("arch")
class _ArchiveOpener(Node):
    def __init__(
        self,
        archive_fn: RawOrVariable[str],
        *,
        query: RawOrVariable[Optional[str]] = None,
        prepare_data: Optional[Callable[["pd.DataFrame"], "pd.DataFrame"]] = None,
        verbose=False,
        keep_going=False,
        print_summary=False,
        encoding="utf-8",
        index_pattern="*ecotaxa_*",
        columns: Optional[List] = None,
    ):
        super().__init__()

        self.archive_fn = archive_fn
        self.query = query
        self.prepare_data = prepare_data
        self.verbose = verbose
        self.keep_going = keep_going
        self.print_summary = print_summary
        self.encoding = encoding
        self.index_pattern = index_pattern
        self.columns = columns

    @staticmethod
    def _fix_types(dataframe):
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
            pd.to_numeric, errors="coerce", axis=1
        )

        return dataframe

    def transform_stream(self, stream: Stream) -> Stream:

        messages = []

        if self.columns is not None:
            usecols = lambda c: any(fnmatch.fnmatch(c, p) for p in self.columns)
        else:
            usecols = None

        stream_estimator = StreamEstimator()

        with closing_if_closable(stream):
            for obj in stream:
                archive_fn, query = self.prepare_input(obj, ("archive_fn", "query"))

                if self.verbose:
                    print(f"EcotaxaReader: Processing archive {archive_fn}...")

                try:
                    archive = Archive(archive_fn)
                except UnknownArchiveError as exc:
                    if self.keep_going:
                        print(exc)
                        messages.append(str(exc))
                        continue
                    raise

                with archive:
                    index_fns = sorted(archive.find(self.index_pattern))

                    with stream_estimator.consume(
                        obj.n_remaining_hint, est_n_emit=len(index_fns)
                    ) as incoming:

                        for index_fn in index_fns:
                            if self.verbose:
                                print(f"EcotaxaReader: Processing index {index_fn}...")

                            index_base = os.path.dirname(index_fn)

                            with archive.read_member(index_fn) as index_fp:
                                dataframe = pd.read_csv(
                                    index_fp,
                                    sep="\t",
                                    low_memory=False,
                                    encoding=self.encoding,
                                    usecols=usecols,
                                )
                                dataframe = self._fix_types(dataframe)

                                if self.prepare_data is not None:
                                    dataframe = self.prepare_data(dataframe)

                                if "img_rank" not in dataframe.columns:
                                    dataframe["img_rank"] = 1

                                if "object_id" not in dataframe.columns:
                                    raise ValueError(
                                        f"object_id missing in {archive_fn}/{index_fn}\n"
                                        f"Columns: {dataframe.columns}"
                                    )

                                if "img_file_name" not in dataframe.columns:
                                    raise ValueError(
                                        f"img_file_name missing in {archive_fn}/{index_fn}\n"
                                        f"Columns: {dataframe.columns}"
                                    )

                                if query is not None:
                                    dataframe = dataframe.query(query)

                                dataframe_by_object_id = dataframe.groupby("object_id")

                                yield self.prepare_output(
                                    obj.copy(),
                                    (
                                        archive_fn,
                                        archive,
                                        index_fn,
                                        index_base,
                                        dataframe_by_object_id,
                                    ),
                                    n_remaining_hint=incoming.emit(),
                                )

            if self.print_summary:
                if messages:
                    print("EcotaxaReader: Messages")
                    for msg in messages:
                        print(msg)


class _ImageLoader:
    def __init__(self, archive: Archive) -> None:
        self.archive = archive

    def load(self, index_base, image_fn) -> IO:
        image_fn = os.path.join(index_base, image_fn)
        image_data = io.BytesIO()
        with self.archive.read_member(image_fn) as image_fp:
            copyfileobj(image_fp, image_data)
        return image_data


@ReturnOutputs
@Output("object")
class _ArchiveObjectReader(Node):
    def __init__(self, arch, *, keep_going, print_summary, image_default_mode) -> None:
        super().__init__()
        self.arch = arch

        self.keep_going = keep_going
        self.print_summary = print_summary
        self.image_default_mode = image_default_mode

    def transform_stream(self, stream: Stream) -> Stream:
        messages = []

        stream_estimator = StreamEstimator()

        # Total number of objects read
        n_objects_read = 0

        with closing_if_closable(stream):
            for obj in stream:
                (
                    archive_fn,
                    archive,
                    index_fn,
                    index_base,
                    dataframe_by_object_id,
                ) = self.prepare_input(obj, "arch")

                with stream_estimator.consume(
                    obj.n_remaining_hint, est_n_emit=len(dataframe_by_object_id)
                ) as incoming:

                    image_loader = _ImageLoader(archive)

                    for i, (object_id, group) in enumerate(dataframe_by_object_id):

                        try:
                            image_data = {
                                row["img_rank"]: image_loader.load(
                                    index_base, row["img_file_name"]
                                )
                                for _, row in group.iterrows()
                            }
                        except Exception as exc:
                            if self.keep_going:
                                print(exc)
                                messages.append(str(exc))
                                continue
                            raise

                        meta = group.iloc[0].to_dict()

                        # Include input metadata
                        meta["morphocut_input_archive"] = archive_fn
                        meta["morphocut_input_index"] = index_fn

                        yield self.prepare_output(
                            obj.copy(),
                            EcotaxaObject(
                                meta=meta,
                                image_data=image_data,
                                index_fn=index_fn,
                                default_mode=self.image_default_mode,
                            ),
                            n_remaining_hint=incoming.emit(),
                        )

                        n_objects_read += 1

            if self.print_summary:
                print(f"EcotaxaReader: Read {n_objects_read} objects.")
                if messages:
                    print("EcotaxaReader: Messages")
                    for msg in messages:
                        print(msg)


@ReturnOutputs
@Output("object")
class EcotaxaReader(Node):
    """
    |stream| Read an archive of images and metadata that is importable to EcoTaxa.

    Args:
        archive_fn (str, Variable): Location of the archive file.
        query (str, Variable, optional): A query string to select a subset of the data.
            See :py:meth:`pandas.DataFrame.query`.
        prepare_data (callable, optional): A function that receives a :py:class:`pandas.DataFrame`
            and returns a modified version.

    Returns:
        object: An :class:`EcotaxaObject` that allows to access image(s) and metadata.

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
                obj = EcotaxaReader("path/to/archive.zip")
            p.transform_stream()
    """

    def __init__(
        self,
        archive_fn: RawOrVariable[str],
        *,
        query: RawOrVariable[Optional[str]] = None,
        prepare_data: Optional[Callable[["pd.DataFrame"], "pd.DataFrame"]] = None,
        verbose=False,
        keep_going=False,
        print_summary=False,
        encoding="utf-8",
        index_pattern="*ecotaxa_*",
        columns: Optional[List] = None,
        image_default_mode=None,
    ):
        super().__init__()
        self.archive_fn = archive_fn
        self.query = query
        self.prepare_data = prepare_data
        self.verbose = verbose
        self.keep_going = keep_going
        self.print_summary = print_summary
        self.encoding = encoding
        self.index_pattern = index_pattern
        self.columns = columns
        self.image_default_mode = image_default_mode

    def transform_stream(self, stream: Stream):
        n_indices_received = 0
        n_indices_processed = 0
        n_objects_received = 0

        n_objects_read = 0

        messages = []

        if self.columns is not None:
            usecols = lambda c: any(fnmatch.fnmatch(c, p) for p in self.columns)
        else:
            usecols = None

        with closing_if_closable(stream):
            stream_estimator = StreamEstimator()
            for obj in stream:
                archive_fn, query = self.prepare_input(obj, ("archive_fn", "query"))

                if self.verbose:
                    print(f"EcotaxaReader: Processing archive {archive_fn}...")

                try:
                    archive = Archive(archive_fn)
                except UnknownArchiveError as exc:
                    if self.keep_going:
                        print(exc)
                        messages.append(str(exc))
                        continue
                    raise

                with archive:
                    index_fns = sorted(archive.find(self.index_pattern))

                    with stream_estimator.consume(
                        obj.n_remaining_hint, est_n_emit=len(index_fns)
                    ) as incoming_archive:

                        n_indices_received += len(index_fns)

                        archive_estimator = StreamEstimator()

                        for index_fn in index_fns:
                            if self.verbose:
                                print(f"EcotaxaReader: Processing index {index_fn}...")

                            index_base = os.path.dirname(index_fn)

                            n_indices_processed += 1

                            with archive.read_member(index_fn) as index_fp:
                                dataframe = pd.read_csv(
                                    index_fp,
                                    sep="\t",
                                    low_memory=False,
                                    encoding=self.encoding,
                                    usecols=usecols,
                                )
                                dataframe = self._fix_types(dataframe)

                                if self.prepare_data is not None:
                                    dataframe = self.prepare_data(dataframe)

                                if "img_rank" not in dataframe.columns:
                                    dataframe["img_rank"] = 1

                                if "object_id" not in dataframe.columns:
                                    raise ValueError(
                                        f"object_id missing in {archive_fn}/{index_fn}\n"
                                        f"Columns: {dataframe.columns}"
                                    )

                                if query is not None:
                                    dataframe = dataframe.query(query)

                                dataframe_by_object_id = dataframe.groupby("object_id")

                                with archive_estimator.consume(
                                    incoming_archive.emit(),
                                    est_n_emit=len(dataframe_by_object_id),
                                ) as incoming_index:

                                    index_estimator = StreamEstimator()

                                    n_objects_received += len(dataframe_by_object_id)

                                    for (object_id, group) in dataframe_by_object_id:
                                        with index_estimator.consume(
                                            incoming_index.emit(), est_n_emit=1
                                        ) as incoming_object:

                                            if "img_file_name" in dataframe.columns:
                                                try:
                                                    image_data = {
                                                        row[
                                                            "img_rank"
                                                        ]: self._load_image(
                                                            archive,
                                                            index_base,
                                                            row["img_file_name"],
                                                        )
                                                        for _, row in group.iterrows()
                                                    }
                                                except MemberNotFoundError as exc:
                                                    if self.keep_going:
                                                        print(exc)
                                                        messages.append(str(exc))
                                                        continue
                                                    raise
                                            else:
                                                image_data = {}

                                            meta = group.iloc[0].to_dict()

                                            yield self.prepare_output(
                                                obj.copy(),
                                                EcotaxaObject(
                                                    meta=meta,
                                                    image_data=image_data,
                                                    index_fn=index_fn,
                                                    default_mode=self.image_default_mode,
                                                    archive_fn=archive_fn,
                                                ),
                                                n_remaining_hint=incoming_object.emit(),
                                            )

                                            n_objects_read += 1

                if self.print_summary:
                    print(f"EcotaxaReader: Read {n_objects_read} objects.")
                    if messages:
                        print("EcotaxaReader: Messages")
                        for msg in messages:
                            print(msg)

    @staticmethod
    def _load_image(archive: Archive, index_base, image_fn):
        image_fn = os.path.join(index_base, image_fn)
        image_data = io.BytesIO()
        with archive.read_member(image_fn) as image_fp:
            copyfileobj(image_fp, image_data)
        return image_data

    @staticmethod
    def _fix_types(dataframe):
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
            pd.to_numeric, errors="coerce", axis=1
        )

        return dataframe
