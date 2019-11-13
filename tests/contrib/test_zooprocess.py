from morphocut.contrib.zooprocess import CalculateZooProcessFeatures
from tests.helpers import BinaryBlobs
from morphocut import Call, Pipeline


def test_CalculateZooProcessFeatures():
    with Pipeline() as p:
        i = FromIterable(range(10))
        mask = BinaryBlobs()
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
