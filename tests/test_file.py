import re

import pytest

from morphocut import Pipeline
from morphocut.file import Find, Glob


@pytest.mark.parametrize("sort", [True, False])
@pytest.mark.parametrize("verbose", [True, False])
def test_Find(data_path, sort, verbose, capsys):
    d = data_path / "images"
    with Pipeline() as pipeline:
        filename = Find(d, [".png"], sort, verbose)

    stream = pipeline.transform_stream()

    filenames = [o[filename] for o in stream]

    if sort:
        assert filenames == sorted(filenames)

    if verbose:
        out = capsys.readouterr().out
        assert re.search(r"^Found \d+ files in .+\.$", out)


def test_Glob(data_path, capsys):
    d = data_path / "images/*.png"
    with Pipeline() as pipeline:
        result = Glob(d, True, prefetch=True, sorted=True)

    pipeline.run()

    # Capture the print output
    captured = capsys.readouterr()

    # Check that matches were found
    assert "matches for" in captured.out

    # Extract the number of matches from the print statement
    matches = int(captured.out.split(" ")[0].replace(",", ""))
    assert matches > 0
