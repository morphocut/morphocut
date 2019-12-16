"""Experiment on processing KOSMOS data using MorphoCut."""

import os

from skimage.util import img_as_ubyte

from morphocut import Call
from morphocut.contrib.ecotaxa import EcotaxaWriter
from morphocut.contrib.zooprocess import CalculateZooProcessFeatures
from morphocut.core import Pipeline
from morphocut.file import Find
from morphocut.image import (
    ExtractROI,
    FindRegions,
    ImageReader,
    ImageWriter,
    RescaleIntensity,
    RGB2Gray,
)
from morphocut.stat import RunningMedian
from morphocut.str import Format
from morphocut.stream import TQDM, Enumerate

import_path = "/home/moi/Work/0-Datasets/Pyrocystis_noctiluca/RAW"
export_path = "/tmp/Pyrocystis_noctiluca"


if __name__ == "__main__":
    print("Processing images under {}...".format(import_path))

    # Create export_path in case it doesn't exist
    os.makedirs(export_path, exist_ok=True)

    # Define processing pipeline
    with Pipeline() as p:
        # Recursively find .jpg files in import_path.
        # Sort to get consective frames.
        abs_path = Find(import_path, [".jpg"], sort=True)

        # Extract name from abs_path
        name = Call(lambda p: os.path.splitext(os.path.basename(p))[0], abs_path)

        # Read image
        img = ImageReader(abs_path)

        # Apply running median to approximate the background image
        flat_field = RunningMedian(img, 10)

        # Correct image
        img = img / flat_field

        # Rescale intensities and convert to uint8 to speed up calculations
        img = RescaleIntensity(img, in_range=(0, 1.1), dtype="uint8")

        # Show progress bar for frames
        TQDM(name)

        # Convert image to uint8 gray
        img_gray = RGB2Gray(img)
        img_gray = Call(img_as_ubyte, img_gray)

        # Apply threshold find objects
        threshold = 0.8  # Call(skimage.filters.threshold_otsu, img_gray)
        mask = img_gray < threshold

        # Write corrected frames
        frame_fn = Format(os.path.join(export_path, "{name}.jpg"), name=name)
        ImageWriter(frame_fn, img)

        # Find objects
        regionprops = FindRegions(mask, img_gray, min_area=100, padding=10)

        # For an object, extract a vignette/ROI from the image
        roi_orig = ExtractROI(img, mask, regionprops, bg_color=255)
        roi_gray = ExtractROI(img_gray, mask, regionprops, bg_color=255)

        # Generate an object identifier
        i = Enumerate()
        object_id = Format("{name}_{i:d}", name=name, i=i)

        # Calculate features
        meta = CalculateZooProcessFeatures(regionprops, prefix="object_")
        meta["object_id"] = object_id

        # Generate object filenames
        orig_fn = Format("{object_id}.jpg", object_id=object_id)
        gray_fn = Format("{object_id}-gray.jpg", object_id=object_id)

        # Write objects to an EcoTaxa archive
        EcotaxaWriter(
            os.path.join(export_path, "pyrocystis.zip"),
            [(orig_fn, roi_orig), (gray_fn, roi_gray)],
            meta,
        )

        # Progress bar for objects
        TQDM(object_id)

    # Execute pipeline
    p.run()
