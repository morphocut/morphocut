"""
Read and write EcoTaxa archives.

    "`EcoTaxa`_ is a web application dedicated to the visual exploration
    and the taxonomic annotation of images that illustrate the
    beauty of planktonic biodiversity."

.. _EcoTaxa: https://ecotaxa.obs-vlfr.fr/
"""
import io
import os
import zipfile

import PIL

from morphocut._optional import import_optional_dependency
from morphocut.graph import Node


class EcotaxaWriter(Node):
    """Zip the image and its meta data."""

    def __init__(
        self, archive_fn, image_fn, image, meta, meta_fn="ecotaxa_export.tsv"
    ):
        super().__init__()
        self.archive_fn = archive_fn

        if not isinstance(image_fn, tuple):
            image_fn = (image_fn, )

        if not isinstance(image, tuple):
            image = (image, )

        if len(image) != len(image_fn):
            raise ValueError("Length of `image` and `image_fn` do not match")

        self.image_fn = image_fn
        self.image = image
        self.meta = meta

        self.meta_fn = meta_fn
        self.image_ext = tuple(os.path.splitext(fn)[1] for fn in self.image_fn)

        self._pd = import_optional_dependency("pandas")

    def transform_stream(self, stream):
        pil_extensions = PIL.Image.registered_extensions()

        with zipfile.ZipFile(self.archive_fn, mode="w") as zf:
            dataframe = []
            for obj in stream:
                image, meta = self.prepare_input(obj, ("image", "meta"))

                for img, img_fn, img_ext in zip(
                    image,
                    self.image_fn,
                    self.image_ext,
                ):
                    pil_format = pil_extensions[img_ext]

                    img = PIL.Image.fromarray(img)
                    img_fp = io.BytesIO()
                    img.save(img_fp, format=pil_format)

                    arcname = img_fn.format(**meta)

                    zf.writestr(arcname, img_fp.getvalue())

                    dataframe.append({**meta, "img_file_name": arcname})

                yield obj

            dataframe = self._pd.DataFrame(dataframe)
            zf.writestr(
                self.meta_fn,
                dataframe.to_csv(sep='\t', encoding='utf-8', index=False)
            )


class EcotaxaReader(Node):
    ...
