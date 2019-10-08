from morphocut.ecotaxa import EcotaxaWriter
from morphocut import Pipeline
from morphocut.io import ImageReader

with Pipeline() as p:
    image = ImageReader(...)
    meta = ...
    EcotaxaWriter("/path/to/archive.zip", "{object_id}.jpg", image, meta)

p.run()
