"""Filesystem-related operations."""

import glob
import os
from pathlib import Path
from typing import Iterable, Set, Union

from morphocut import Node, Output, RawOrVariable, ReturnOutputs, closing_if_closable
from morphocut.core import Stream
from morphocut.utils import StreamEstimator

__all__ = ["Find", "Glob"]


@ReturnOutputs
@Output("abs_path")
class Find(Node):
    """
    |stream| Find files under the specified directory.

    Args:
        root (str or Path, raw or Variable): Root path where images should be found.
        extensions (list, raw): List of allowed extensions (including the leading dot).
        sort (bool): Sort results alphabetically (default false).
        verbose (bool): Print info about files (default false).

    Returns:
        Variable[str]: Path of the matching file.
    """

    def __init__(
        self,
        root: RawOrVariable[Union[str, Path]],
        extensions: Iterable,
        sort=False,
        verbose=False,
    ):
        super().__init__()

        self.root = root
        self.extensions = set(extensions)  # type: Set[str]
        self.sort = sort
        self.verbose = verbose

    def transform_stream(self, stream):
        with closing_if_closable(stream):
            for obj in stream:
                root = self.prepare_input(obj, "root")

                # Convert to str to allow Path objects in Python 3.5
                root = str(root)

                for root, dirnames, filenames in os.walk(root):
                    if self.sort:
                        dirnames[:] = sorted(dirnames)
                        filenames = sorted(filenames)

                    if self.verbose:
                        print(f"Found {len(filenames):,d} files in {root}.")
                    for fn in filenames:
                        ext = os.path.splitext(fn)[1]

                        # Skip non-allowed extensions
                        if ext not in self.extensions:
                            continue

                        yield self.prepare_output(obj.copy(), os.path.join(root, fn))


@ReturnOutputs
@Output("path")
class Glob(Node):
    """
    |stream| Find files matching ``pathname``.

    For more information see :py:mod:`glob`.

    Args:
        pathname (str or Variable[str]): Pattern for path names containing a path specification.
            Can be either absolute (like ``/path/to/file``) or relative (like ``../../foo/*/*.bar``)
            and can contain shell-style wildcards.
            If the pattern is followed by an ``os.sep`` or ``os.altsep`` then files will not match.
        recursive (bool or Variable[bool]): If true, the pattern "``**``"
            will match any files and zero or more directories, subdirectories
            and symbolic links to directories.

    Returns:
        Variable[str]: Path matching ``pathname``.
    """

    def __init__(
        self,
        pathname: RawOrVariable[str],
        recursive: RawOrVariable[bool] = False,
        prefetch=False,
    ):
        super().__init__()

        self.pathname = pathname
        self.recursive = recursive
        self.prefetch = prefetch

    def transform_stream(self, stream: Stream):
        with closing_if_closable(stream):

            est = StreamEstimator()

            for obj in stream:
                pathname, recursive = self.prepare_input(obj, ("pathname", "recursive"))

                # Convert to str to allow Path objects in Python 3.5
                pathname = str(pathname)

                matches = glob.iglob(pathname, recursive=recursive)

                est_n_emit = None
                if self.prefetch:
                    matches = list(matches)
                    est_n_emit = len(matches)
                    print(f"{est_n_emit} matches for {pathname}")

                with est.consume(
                    obj.n_remaining_hint, est_n_emit=est_n_emit
                ) as incoming:
                    for path in matches:
                        yield self.prepare_output(
                            obj.copy(), path, n_remaining_hint=incoming.emit()
                        )
