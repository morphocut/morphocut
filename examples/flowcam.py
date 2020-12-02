"""Process FlowCam data using MorphoCut and store as EcoTaxa archive."""

import os

from morphocut import Pipeline
from morphocut.contrib.ecotaxa import EcotaxaWriter
from morphocut.contrib.zooprocess import CalculateZooProcessFeatures
from morphocut.file import Find
from morphocut.image import ImageProperties, RGB2Gray
from morphocut.integration.flowcam import FlowCamReader
from morphocut.str import Format
from morphocut.stream import TQDM

import_path = "../tests/data/flowcam"
export_path = "/tmp/flowcam"

os.makedirs(export_path, exist_ok=True)

if __name__ == "__main__":
    print("Processing images under {}...".format(import_path))

    # All operations are inserted into a pipeline and are only executed when Pipeline.run() is called.
    with Pipeline() as p:
        # Find .lst files in import_path
        lst_fn = Find(import_path, [".lst"])

        # Display a progress indicator for the .lst files
        TQDM(lst_fn)

        # Read objects from a .lst file
        obj = FlowCamReader(lst_fn)

        # Extract object image and convert to graylevel
        img = obj.image
        img_gray = RGB2Gray(img, True)

        # Extract object mask
        mask = obj.mask

        # Extract metadata from the FlowCam
        object_meta = obj.data

        # Construct object ID
        object_id = Format(
            "{lst_name}_{id}", lst_name=obj.lst_name, _kwargs=object_meta
        )
        object_meta["id"] = object_id

        # Calculate object properties (area, eccentricity, equivalent_diameter, mean_intensity, ...). See skimage.measure.regionprops.
        regionprops = ImageProperties(mask, img_gray)
        # Append object properties to metadata in a ZooProcess-like format
        object_meta = CalculateZooProcessFeatures(regionprops, object_meta)

        # Write each object to an EcoTaxa archive.
        # Here, three different versions are written. Remove what you do not need.
        EcotaxaWriter(
            os.path.join(export_path, "export.zip"),
            [
                # The original RGB image
                (Format("{object_id}.jpg", object_id=object_id), img),
                # A graylevel version
                (Format("{object_id}_gray.jpg", object_id=object_id), img_gray),
                # The binary mask
                (Format("{object_id}_mask.jpg", object_id=object_id), mask),
            ],
            object_meta=object_meta,
        )

        # Display progress indicator for individual objects
        TQDM(object_id)

    p.run()
