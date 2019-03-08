"""
Processing nodes are generators.
"""

from morphocut.processing.pipeline.base import NodeBase, SimpleNodeBase
from morphocut.processing.pipeline.input import LocalDirectoryInput
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
from morphocut.processing.pipeline.image_manipulator import GreyImage, WhiteBackgroundImage, ContourImage
