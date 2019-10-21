import glob
import os
from typing import Collection

from morphocut import Node, Output, RawOrVariable, ReturnOutputs, Variable


@ReturnOutputs
@Output("abs_path")
class Find(Node):
    """
    Find files under the specified directory.

    Args:
        root (str): Root path where images should be found.
        extensions (list): List of allowed extensions (including the leading dot).

    Returns:
        Variable[str]: Path of the matching file.
    """

    def __init__(self, root: str, extensions: Collection):
        super().__init__()

        self.root = root
        self.extensions = set(extensions)

    def transform_stream(self, stream):
        for obj in stream:
            root = self.prepare_input(obj, "root")
            for root, _, filenames in os.walk(root):
                for fn in filenames:
                    ext = os.path.splitext(fn)[1]

                    # Skip non-allowed extensions
                    if ext not in self.extensions:
                        continue

                    yield self.prepare_output(
                        {},
                        os.path.join(root, fn),
                    )


@ReturnOutputs
@Output("path")
class Glob(Node):
    """
    Find files matching ``pathname``.

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

    def __init__(self, pathname: RawOrVariable[str], recursive: RawOrVariable[bool] = False):
        super().__init__()

        self.pathname = pathname
        self.recursive = recursive

    def transform_stream(self, stream):
        for obj in stream:
            pathname, recursive = self.prepare_input(
                obj, ("pathname", "recursive"))
            for path in glob.iglob(pathname, recursive=recursive):
                yield self.prepare_output(
                    {},
                    path
                )
