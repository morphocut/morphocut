import csv
import os
from typing import List, Mapping, Optional, Sequence, Callable, Any, Iterable

from morphocut import Node, Output, RawOrVariable, ReturnOutputs
from morphocut._optional import import_optional_dependency


@ReturnOutputs
class ToDataFrame(Node):
    """
    Create a DataFrame from stream contents.
    
    Args:
        _after_stream (Callable): Callback that receives the DataFrame  after finishing the processing.
        *args (Mapping or Variable): Mappings of column/value.
        _columns: (Collection, optional): Restrict columns in *args.
        **kwargs (Any or Variable): Additional values.
    """

    def __init__(
        self,
        _after_stream: Callable,
        *args: RawOrVariable[Mapping],
        _columns: Optional[Iterable] = None,
        **kwargs: RawOrVariable[Any],
    ):
        super().__init__()

        self._after_stream = _after_stream
        self.args = args
        if _columns is not None:
            _columns = set(_columns)
        self._columns = _columns
        self.kwargs = kwargs
        self.dataframe = []  # type: List[Mapping]

        self._pd = import_optional_dependency("pandas")

    def transform(self, args, kwargs):
        row = {}
        for mapping in args:
            if self._columns is not None:
                row.update({k: v for k, v in mapping.items() if k in self._columns})

        row.update(kwargs)

        self.dataframe.append(row)

    def after_stream(self):
        dataframe = self._pd.DataFrame(self.dataframe)
        self._after_stream(dataframe)


@ReturnOutputs
class ToCSV(ToDataFrame):
    def __init__(
        self,
        path_or_buf,
        *args: RawOrVariable[Mapping],
        _columns: Optional[Iterable] = None,
        _drop_duplicates: Optional[Iterable] = None,
        _to_csv_kwargs: Optional[dict] = None,
        **kwargs: RawOrVariable[Any],
    ):
        super().__init__(None, *args, _columns=_columns, **kwargs)

        self.path_or_buf = path_or_buf
        self._drop_duplicates = _drop_duplicates

        if _to_csv_kwargs is None:
            _to_csv_kwargs = {}

        _to_csv_kwargs.setdefault("index", False)

        self._to_csv_kwargs = _to_csv_kwargs

    def after_stream(self):
        dataframe = self._pd.DataFrame(self.dataframe)

        if self._drop_duplicates is not None:
            dataframe.drop_duplicates(subset=self._drop_duplicates, inplace=True)

        dataframe.to_csv(self.path_or_buf, **self._to_csv_kwargs)


@ReturnOutputs
@Output("meta_out")
class JoinMetadata(Node):
    """Join information from a CSV/TSV/Excel/... file."""

    def __init__(
        self,
        filename: str,
        data: RawOrVariable[Mapping] = None,
        on=None,
        fields: Sequence = None,
    ):
        super().__init__()

        self.data = data
        self.on = on

        pd = import_optional_dependency("pandas")

        ext = os.path.splitext(filename)[1]

        if ext in (".xls", ".xlsx"):
            dataframe = pd.read_excel(filename, usecols=fields)
        else:
            with open("example.csv", newline="") as csvfile:
                dialect = csv.Sniffer().sniff(csvfile.read(1024))
            dataframe = pd.read_csv(filename, dialect=dialect, usecols=fields)

        dataframe.set_index(self.on, inplace=True, verify_integrity=True)

        self.dataframe = dataframe

    def transform(self, data):
        if data is None:
            data = {}

        key = data[self.on]

        row = self.dataframe.loc[key].to_dict()

        return {**data, **row}
