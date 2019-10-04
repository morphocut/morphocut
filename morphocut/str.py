from morphocut._optional import import_optional_dependency
from morphocut.graph import Node, Output


@Output("string")
class Format(Node):

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
        return fmt.format(*_args, *args, **_kwargs, **kwargs)


@Output("meta")
class Parse(Node):
    """Parse information from a path.

    Args:
        pattern (str): The pattern to look for in the input.
        case_sensitive (bool): Match pattern with case.
    """

    def __init__(self, pattern: str, string, case_sensitive: bool = False):
        super().__init__()

        self.string = string

        parse = import_optional_dependency("parse")

        @parse.with_pattern(".*")
        def parse_greedystar(text):
            return text

        extra_types = {"greedy": parse_greedystar}

        self.pattern = parse.compile(
            pattern, extra_types=extra_types, case_sensitive=case_sensitive
        )

    def transform(self, string):
        try:
            result = self.pattern.parse(string)
        except TypeError:
            print(repr(string))
            raise

        if result is None:
            raise ValueError(
                "No match for {} in {}".format(self.pattern._format, string)
            )

        return result.named
