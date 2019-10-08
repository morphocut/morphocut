"""Experiment on processing KOSMOS data using MorphoCut."""
import os

import numpy as np
import skimage
import skimage.io
import skimage.measure
import skimage.segmentation

from morphocut import LambdaNode, Pipeline
from morphocut.ecotaxa import EcotaxaWriter
from morphocut.file import Find
from morphocut.image import ExtractROI, FindRegions, Rescale, ThresholdConst
from morphocut.pandas import JoinMetadata, PandasWriter
from morphocut.str import Format, Parse
from morphocut.stream import TQDM, Enumerate, PrintObjects, StreamBuffer
from morphocut.zooprocess import CalculateZooProcessFeatures

#import_path = "/data-ssd/mschroeder/Datasets/generic_zooscan_peru_kosmos_2017"
import_path = "/home/moi/Work/0-Datasets/generic_zooscan_peru_kosmos_2017"
#import_path = '/studi-tmp/mkhan/generic_zooscan_peru_kosmos_2017'

if __name__ == "__main__":
    image_root = os.path.join(import_path, "raw")
    print("Processing images under {}...".format(image_root))

    with Pipeline() as p:
        # Images are named <sampleid>/<anything>_<a|b>.tif
        # e.g. generic_Peru_20170226_slow_M1_dnet/Peru_20170226_M1_dnet_1_8_a.tif
        abs_path = Find(image_root, [".tif"])()

        rel_path = LambdaNode(os.path.relpath, abs_path, image_root)()
        meta = Parse(
            "generic_{sample_id}/{:greedy}_{sample_split:d}_{sample_nsplit:d}_{sample_subid}.tif",
            rel_path
        )()

        meta = JoinMetadata(
            os.path.join(
                import_path, "Morphocut_header_scans_peru_kosmos_2017.xlsx"
            ),
            meta,
            "sample_id",
        )()

        PandasWriter(
            os.path.join(import_path, "meta.csv"),
            meta,
            drop_duplicates_subset="sample_id",
        )()

        img = LambdaNode(skimage.io.imread, abs_path)()

        StreamBuffer(maxsize=2)

        TQDM(rel_path)()

        img = Rescale(img, in_range=(9252, 65278), dtype=np.uint8)()

        mask = ThresholdConst(img, 245)()  # 245(ubyte) / 62965(uint16)
        mask = LambdaNode(skimage.segmentation.clear_border, mask)()

        regionprops = FindRegions(mask, img, 100, padding=10)()

        # TODO: Draw object info

        # Extract a vignette from the image
        vignette = ExtractROI(img, regionprops)()

        # # Extract features from vignette
        # model = resnet18(pretrained=True)
        # model = torch.nn.Sequential(OrderedDict(
        #     list(model.named_children())[:-2]))

        # features = PyTorch(lambda x: model(x).cpu().numpy())(vignette)

        i = Enumerate()()
        object_id = Format(
            "{sample_id}_{sample_split:d}_{sample_nsplit:d}_{sample_subid}_{i:d}",
            _kwargs=meta,
            i=i,
        )()
        meta["object_id"] = object_id
        meta = CalculateZooProcessFeatures(regionprops, meta, "object_")()

        # EcotaxaWriter(
        #     os.path.join(import_path, "export.zip"), "{object_id}.jpg",
        #     vignette, meta
        # )()

        TQDM(object_id)()

    p.run()
