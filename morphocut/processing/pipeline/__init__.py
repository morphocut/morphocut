"""
Processing nodes are generators.
"""

from morphocut.processing.pipeline.base import NodeBase, SimpleNodeBase
from morphocut.processing.pipeline.pipeline import Pipeline, MultiThreadPipeline
from morphocut.processing.pipeline.processor import Processor
from morphocut.processing.pipeline.dataloader import DataLoader
from morphocut.processing.pipeline.exporter import Exporter
from morphocut.processing.pipeline.progress import Progress
from morphocut.processing.pipeline.vignette_corrector import VignetteCorrector
# from morphocut.processing.pipeline.threshold_otsu import ThresholdOtsu
# from morphocut.processing.pipeline.extract_regions import ExtractRegions
# from morphocut.processing.pipeline.color import GrayToRGB
# from morphocut.processing.pipeline.contour import ContourTransform
from morphocut.processing.pipeline.color import BGR2Gray
from morphocut.processing.pipeline.threshold_otsu import ThresholdOtsu
from morphocut.processing.pipeline.extract_regions import ExtractRegions
from morphocut.processing.pipeline.annotation import FadeBackground, DrawContours
from morphocut.processing.pipeline.debug import PrintFacettes


def get_default_pipeline(import_path, export_path):
    return Pipeline([
        DataLoader(import_path, output_facet="raw"),
        Progress("Loaded"),
        VignetteCorrector(input_facet="raw", output_facet="color"),
        BGR2Gray(input_facet="color", output_facet="gray"),
        ThresholdOtsu(input_facet="gray", output_facet="mask"),
        ExtractRegions(
            mask_facet="mask",
            intensity_facet="gray",
            image_facets=["color", "gray", "mask"],
            output_facet="features",
            padding=10),
        FadeBackground(
            image_facet="gray",
            mask_facet="mask",
            output_facet="bg_white", alpha=1),
        DrawContours(
            image_facet="color",
            mask_facet="mask",
            output_facet="color_contours"),
        Exporter(
            export_path,
            data_facets=["features"],
            img_facets=["bg_white", "color", "color_contours"])
    ])
