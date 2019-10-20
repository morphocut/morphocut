from ._version import get_versions
from .core import LambdaNode, Node, Output, Pipeline, ReturnOutputs

__version__ = get_versions()['version']
del get_versions
