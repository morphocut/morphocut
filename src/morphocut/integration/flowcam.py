"""
Read FlowCam® collage files.

    `FlowCam®`_ is an automated particle analysis instrument for measuring size
    and shape of microscopic particles in a fluid medium.

.. _FlowCam®: https://www.fluidimaging.com/
"""
import collections.abc
import csv
import itertools
import operator
import os.path
import pathlib
from typing import Union

import dateutil.parser
import numpy as np
import PIL.Image

from morphocut import Node, Output, RawOrVariable, ReturnOutputs, closing_if_closable

_DTYPES = {
    "int32": int,
    "double": float,  # lambda x: float("nan") if x is None else float(x),
    "string": str,  # lambda x: "" if x is None else str(x),
    "guid": str,
    "timestamp": dateutil.parser.parse,
}


class _LstReader(collections.abc.Iterable):
    def __init__(self, lst_fn):
        self.lst_fn = lst_fn

    def __iter__(self):
        with open(self.lst_fn, "r") as f:
            version = next(f).strip()

            if version != "017":
                raise ValueError("Unrecognized version string: {}".format(version))

            num_fields_name, num_fields = next(f).strip().split("|", 1)
            if num_fields_name != "num-fields":
                raise ValueError("Expected num-fields, got {}".format(num_fields_name))

            num_fields = int(num_fields)

            fields = []
            for _ in range(num_fields):
                field, dtype = next(f).strip().split("|", 1)
                dtype = _DTYPES[dtype]
                fields.append((field, dtype))

            # Iterate over the remaining lines
            reader = csv.DictReader(f, fieldnames=[f[0] for f in fields], delimiter="|")

            for row in reader:
                row = {field: dtype(row[field]) for field, dtype in fields}
                yield row


class FlowCamObject:
    """
    A single object.

    Not to be instanciated manually.
    
    .. seealso::
         :py:class:`~FlowCamReader`
    """

    def __init__(self, data, lst_name, collage, collage_bin):
        self.data = data
        self.lst_name = lst_name
        self.collage = collage
        self.collage_bin = collage_bin

    def __getattr__(self, name):
        try:
            return self.data[name]
        except KeyError:
            raise AttributeError(name)

    @property
    def slice(self):
        return (
            slice(self.image_y, self.image_y + self.image_h),
            slice(self.image_x, self.image_x + self.image_w),
        )

    @property
    def image(self):
        """The object image."""
        return self.collage[self.slice]

    @property
    def mask(self):
        """The object mask."""
        return self.collage_bin[self.slice]


@ReturnOutputs
@Output("regionprops")
class FlowCamReader(Node):
    """
    |stream| Read a flowcam sample with collage files.

    .. note::
        This Node creates multiple objects per incoming object.

    Args:
        lst_fn (str or Path, optional): The path to a ``.lst`` file.

    Example:
        .. code-block:: python

            obj = FlowCamReader("flowcam.lst")
            image = obj.image
            mask = obj.mask
    """

    def __init__(self, lst_fn: RawOrVariable[Union[str, pathlib.Path]]):
        super().__init__()

        self.lst_fn = lst_fn

    def transform_stream(self, stream):
        with closing_if_closable(stream):
            for obj in stream:
                lst_fn = self.prepare_input(obj, "lst_fn")

                # Convert to str to allow Path objects
                lst_fn = str(lst_fn)

                root_path, lst_name = os.path.split(lst_fn)
                lst_name = os.path.splitext(lst_name)[0]

                reader = _LstReader(lst_fn)

                for collage_file, data in itertools.groupby(
                    reader, operator.itemgetter("collage_file")
                ):
                    # Load image collage
                    collage_fn = os.path.join(root_path, collage_file)
                    collage = np.array(PIL.Image.open(collage_fn))

                    # Load bin collage
                    base, ext = os.path.splitext(collage_file)
                    collage_bin_fn = os.path.join(
                        root_path, "{}_bin{}".format(base, ext)
                    )
                    collage_bin = np.array(PIL.Image.open(collage_bin_fn)).astype(bool)

                    for row in data:
                        yield self.prepare_output(
                            obj.copy(),
                            FlowCamObject(row, lst_name, collage, collage_bin),
                        )

