from morphocut import Pipeline, LambdaNode
from morphocut.contrib.ecotaxa import EcotaxaReader, EcotaxaWriter
from morphocut.stream import FromIterable
from tests.helpers import BinaryBlobs, Const
from morphocut.str import Format


def test_ecotaxa(tmp_path):
    archive_fn = tmp_path / "ecotaxa.zip"
    print(archive_fn)

    # Create an archive
    with Pipeline() as p:
        i = FromIterable(range(10))

        meta = LambdaNode(dict, i=i, foo="Sömé UTF-8 ſtríng…")
        image = BinaryBlobs()
        image_name = Format("image_{}", i)

        EcotaxaWriter(archive_fn, image, image_name, meta)

    result = [o.to_dict() for o in p.transform_stream()]

    # Read the archive
    with Pipeline() as p:
        EcotaxaReader(archive_fn)

    roundtrip_result = [o.to_dict() for o in p.transform_stream()]

    assert result == roundtrip_result
