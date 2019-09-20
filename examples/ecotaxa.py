from morphocut.ecotaxa import ArchiveWriter
from morphocut.graph import Pipeline
from morphocut.io import ImageReader

with Pipeline() as p:
    image = ImageReader(...)
    meta = ...
    ArchiveWriter("/path/to/archive.zip", image, meta)

p.run()
