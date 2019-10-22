from ._version import get_versions
from .core import (LambdaNode, Node, Output, Pipeline, RawOrVariable,
                   ReturnOutputs, Variable)

__version__ = get_versions()['version']
del get_versions
