"""
Processing nodes are generators.
"""

from morphocut.processing.pipeline.base import NodeBase
from morphocut.processing.pipeline.input import LocalDirectoryInput
from morphocut.processing.pipeline.pipeline import Pipeline, MultiThreadPipeline
from morphocut.processing.pipeline.processor import Processor
from morphocut.processing.pipeline.dataloader import DataLoader
from morphocut.processing.pipeline.exporter import Exporter
from morphocut.processing.pipeline.progress import Progress
from morphocut.processing.pipeline.vignette_corrector import VignetteCorrector
from morphocut.processing.pipeline.image_manipulator import ImageManipulator, GreyImage, ContourImage, WhiteBackgroundImage
