from morphocut import Pipeline
from morphocut.str import Parse


#TODO: Not working. Still under development
def test_Parse():
    pattern = "This is a {}"
    string = "This is a TEST"
    case_sensitive = False

    with Pipeline() as pipeline:
        result = Parse(pattern, string, case_sensitive)()

    stream = pipeline.transform_stream()
    obj = next(stream)

    assert obj[result] == "TEST"