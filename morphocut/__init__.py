from ._version import get_versions
from .core import LambdaNode, Node, Output, Pipeline

__version__ = get_versions()['version']
del get_versions
