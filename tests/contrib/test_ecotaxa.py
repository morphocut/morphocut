import tarfile
import zipfile
from numpy.testing import assert_equal

from morphocut import Call, Pipeline
from morphocut.contrib.ecotaxa import EcotaxaReader, EcotaxaWriter
from morphocut.str import Format
from morphocut.stream import Unpack
from tests.helpers import BinaryBlobs

import pytest


@pytest.mark.parametrize("ext", [".tar", ".zip"])
def test_ecotaxa(tmp_path, ext):
    archive_fn = str(tmp_path / ("ecotaxa" + ext))
    print(archive_fn)

    # Create an archive
    with Pipeline() as p:
        i = Unpack(range(10))

        object_id = Format("foo{:d}", i)

        meta = Call(dict, object_id=object_id, i=i, foo="Sömé UTF-8 ſtríng…")
        image = BinaryBlobs()
        image_name = Format("image_{}.png", i)

        EcotaxaWriter(
            archive_fn,
            (image_name, image),
            meta,
            object_meta={"foo": 0},
            acq_meta={"foo": 1},
            process_meta={"foo": 2},
            sample_meta={"foo": 3},
        )

    # Execute pipeline and collect results
    result = [o.to_dict(meta=meta, image=image) for o in p.transform_stream()]

    if ext == ".zip":
        assert zipfile.is_zipfile(archive_fn), f"{archive_fn} is not a zip file"
    elif ext == ".tar":
        assert tarfile.is_tarfile(archive_fn), f"{archive_fn} is not a tar file"

    # Read the archive
    with Pipeline() as p:
        obj = EcotaxaReader(archive_fn)

    roundtrip_result = [o.to_dict(obj=obj) for o in p.transform_stream()]

    for meta_field in ("i", "foo"):
        assert [o["meta"][meta_field] for o in result] == [
            o["obj"].meta[meta_field] for o in roundtrip_result
        ]

    for i, prefix in enumerate(("object_", "acq_", "process_", "sample_")):
        assert [o["meta"][prefix + "foo"] for o in result] == [
            i for _ in roundtrip_result
        ]

    assert_equal(
        [o["image"] for o in result], [o["obj"].image for o in roundtrip_result]
    )
