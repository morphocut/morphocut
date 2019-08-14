"""
Processing nodes are generators.
"""

from morphocut.pipeline.base import NodeBase, SimpleNodeBase
from morphocut.pipeline.pipeline import Pipeline, MultiThreadPipeline
from morphocut.pipeline.dataloader import DataLoader
from morphocut.pipeline.exporter import Exporter
from morphocut.pipeline.progress import Progress
from morphocut.pipeline.vignette_corrector import VignetteCorrector
from morphocut.pipeline.color import RGB2Gray, Gray2RGB
from morphocut.pipeline.threshold_otsu import ThresholdOtsu
from morphocut.pipeline.extract_regions import ExtractRegions
from morphocut.pipeline.annotation import FadeBackground, DrawContours
from morphocut.pipeline.debug import PrintFacettes
from morphocut.pipeline.object_scale import ObjectScale
