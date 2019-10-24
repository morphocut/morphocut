import warnings
from typing import Mapping, Optional, Tuple

from morphocut import Node, Output, RawOrVariable, ReturnOutputs, Variable
from morphocut._optional import import_optional_dependency


@ReturnOutputs
@Output("string")
class Format(Node):
    """
    Format a string just like :py:meth:`str.format`.

    This class allows you to create and customize your own string formatting behaviors using the same 
    implementation as the :py:meth:`str.format` methods.
    
    Args:
        fmt (str): A format in which we want our string to be.
        *args: Arguments to be replaced with placeholders in fmt
        _args: Arguments to be appended after *args
        _kwargs: Key value paired arguments
        **kwargs: Key value paired arguments to be appended after _kwargs

    As positional arguments, :py:meth:`str.format` receives ``args`` then ``_args``.
    As keyword arguments, :py:meth:`str.format` receives ``_kwargs`` then ``kwargs``.
    This means that keys passed as keyword arguments overwrite keys in a dict passed as ``_kwargs``.

    Example:
        .. code-block:: python
            
            fmt = "{},{},{},{},{},{},{a},{b},{c},{d}"
            args = (1, 2, 3)
            _args = (4, 5, 6)
            _kwargs = {"a": 7, "b": 8}
            kwargs = {"c": 9, "d": 10}

            with Pipeline() as pipeline:
                result = Format(fmt, *args, _args=_args, _kwargs=_kwargs, **kwargs)

    """

    def __init__(self, fmt: RawOrVariable[str], *args: Tuple[RawOrVariable], _args: Optional[RawOrVariable[Tuple]] = None, _kwargs: RawOrVariable[Mapping] = None, **kwargs: Mapping[str, RawOrVariable]):
        super().__init__()
        self.fmt = fmt
        self._args = _args or ()
        self._kwargs = _kwargs or {}
        self.args = args
        self.kwargs = kwargs

    def transform(
        self, fmt: str, _args: tuple, _kwargs: dict, args: tuple, kwargs: dict
    ):
        kwargs = {**_kwargs, **kwargs}
        return fmt.format(*args, *_args, **kwargs)


class ParseWarning(UserWarning):
    """Issued by :py:class:`Parse`."""


@ReturnOutputs
@Output("meta")
class Parse(Node):
    """
    Parse information from a string.

    Parse strings using a specification based on the :ref:`Python Format String Syntax <python:formatstrings>`.

    .. note::
        The external dependency `parse`_ is required to use this Node.

    .. _parse: https://github.com/r1chardj0n3s/parse

    Args:
        fmt (str): The pattern to look for in the input.
        string (str): A string input which is to be parsed
        case_sensitive (bool): Match pattern with case.

    Example:
        .. code-block:: python

            fmt = "This is a {named}"
            string = "This is a TEST"
            case_sensitive = True
            
            with Pipeline() as pipeline:
                result = Parse(fmt, string, case_sensitive)

    """

    def __init__(self, fmt: RawOrVariable[str], string: RawOrVariable, case_sensitive: bool = False):
        super().__init__()

        self.fmt = fmt
        self.string = string
        self.case_sensitive = case_sensitive

        self._parse = import_optional_dependency("parse")

        @self._parse.with_pattern(".*")
        def parse_greedystar(text):
            return text

        self._extra_types = {"greedy": parse_greedystar}

        if not isinstance(fmt, Variable):
            self._parser = self._compile(fmt)
        else:
            self._parser = None

    def _compile(self, fmt):
        parser = self._parse.compile(
            fmt,
            extra_types=self._extra_types,
            case_sensitive=self.case_sensitive
        )
        if not parser._named_fields:
            warnings.warn(
                "Pattern does not include any named fields: {}".format(fmt),
                ParseWarning,
            )
        return parser

    def transform(self, fmt, string):
        parser = self._parser

        if parser is None:
            parser = self._compile(fmt)

        result = parser.parse(string)

        if result is None:
            raise ValueError(
                "No match for {} in {}".format(self._parser._format, string)
            )

        return result.named
