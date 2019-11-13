from numpy.testing import assert_equal

from morphocut import Call, Pipeline
from morphocut.contrib.ecotaxa import EcotaxaReader, EcotaxaWriter
from morphocut.str import Format
from morphocut.stream import FromIterable
from tests.helpers import BinaryBlobs


def test_ecotaxa(tmp_path):
    archive_fn = tmp_path / "ecotaxa.zip"
    print(archive_fn)

    # Create an archive
    with Pipeline() as p:
        i = FromIterable(range(10))

        meta = Call(dict, i=i, foo="Sömé UTF-8 ſtríng…")
        image = BinaryBlobs()
        image_name = Format("image_{}", i)

        EcotaxaWriter(archive_fn, image, image_name, meta, image_ext=".png")

    result = [o.to_dict(meta=meta, image=image) for o in p.transform_stream()]

    # Read the archive
    with Pipeline() as p:
        image, meta = EcotaxaReader(archive_fn)

    roundtrip_result = [o.to_dict(meta=meta, image=image) for o in p.transform_stream()]

    for meta_field in ("i", "foo"):
        assert [o["meta"][meta_field] for o in result] == [
            o["meta"][meta_field] for o in roundtrip_result
        ]

    assert_equal([o["image"] for o in result], [o["image"] for o in roundtrip_result])
