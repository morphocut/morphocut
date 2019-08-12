"""
Processing nodes are generators.
"""

from morphocut.processing.pipeline.base import NodeBase, SimpleNodeBase, LogBase
from morphocut.processing.pipeline.pipeline import Pipeline, MultiThreadPipeline
from morphocut.processing.pipeline.processor import Processor
from morphocut.processing.pipeline.dataloader import DataLoader
from morphocut.processing.pipeline.exporter import Exporter
from morphocut.processing.pipeline.progress import Progress
from morphocut.processing.pipeline.job_progress import JobProgress
from morphocut.processing.pipeline.vignette_corrector import VignetteCorrector
from morphocut.processing.pipeline.color import BGR2Gray
from morphocut.processing.pipeline.threshold_otsu import ThresholdOtsu
from morphocut.processing.pipeline.extract_regions import ExtractRegions
from morphocut.processing.pipeline.annotation import FadeBackground, DrawContours
from morphocut.processing.pipeline.debug import PrintFacettes
from morphocut.processing.pipeline.object_scale import ObjectScale
from morphocut.processing.pipeline.logs import ObjectCountLog, ParamsLog
from morphocut.processing.pipeline.save_metadata import SaveMetadata


def get_default_pipeline(import_path, export_path):
    return Pipeline([
        DataLoader(import_path, output_facet="raw"),
        JobProgress(),
        Progress("Loaded"),
        VignetteCorrector(input_facet="raw", output_facet="color"),
        BGR2Gray(input_facet="color", output_facet="gray"),
        ThresholdOtsu(input_facet="gray", output_facet="mask"),
        ExtractRegions(
            mask_facet="mask",
            intensity_facet="gray",
            image_facets=["color", "gray", "mask"],
            output_facet="roi",
            padding=10,
            min_area=30),
        FadeBackground(
            image_facet="gray",
            mask_facet="roi",
            output_facet="bg_white", alpha=1),
        DrawContours(
            image_facet="color",
            mask_facet="roi",
            output_facet="color_contours"),
        ObjectScale(input_facets=["color_contours"],
                    output_facets=["color_contours_scale"]),
        Exporter(
            export_path,
            data_facets=["roi"],
            img_facets=["bg_white", "color", "color_contours", "color_contours_scale"])
    ])


def get_default_pipeline_parameterized(import_path, export_path, params):
    input_file_logger = ObjectCountLog('input_files')
    output_file_logger = ObjectCountLog('extracted_segments')
    param_logger = ParamsLog('parameters', params)

    nodes = [
        DataLoader(import_path, output_facet="raw", **params['DataLoader']),
        JobProgress(),
        Progress("Loaded"),
        input_file_logger,
        VignetteCorrector(input_facet="raw", output_facet="color",
                          **params['VignetteCorrector']),
        BGR2Gray(input_facet="color", output_facet="gray"),
        ThresholdOtsu(input_facet="gray", output_facet="mask"),
        ExtractRegions(
            mask_facet="mask",
            intensity_facet="gray",
            image_facets=["color", "gray", "mask"],
            output_facet="roi",
            **params['ExtractRegions']),
        FadeBackground(
            image_facet="gray",
            mask_facet="roi",
            output_facet="bg_white",
            **params['FadeBackground']),
        DrawContours(
            image_facet="color",
            mask_facet="roi",
            output_facet="color_contours",
            **params['DrawContours']),
        ObjectScale(input_facets=["color_contours"],
                    output_facets=["color_contours_scale"],
                    **params['ObjectScale']),
        output_file_logger,
        SaveMetadata(loggers=[input_file_logger,
                              output_file_logger, param_logger]),
        Exporter(
            export_path,
            data_facets=["roi"],
            loggers=[input_file_logger, output_file_logger, param_logger],
            **params['Exporter'])
    ]

    return Pipeline(nodes)
