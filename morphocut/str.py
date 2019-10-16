from morphocut._optional import import_optional_dependency
from morphocut import Node, Output
from morphocut.core import Variable
import warnings


@Output("string")
class Format(Node):
    """Format a string just like :py:meth:`str.format`."""

    def __init__(self, fmt, *args, _args=None, _kwargs=None, **kwargs):
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


@Output("meta")
class Parse(Node):
    """Parse information from a path.

    Args:
        fmt (str): The pattern to look for in the input.
        case_sensitive (bool): Match pattern with case.
    """

    def __init__(self, fmt, string, case_sensitive: bool = False):
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
