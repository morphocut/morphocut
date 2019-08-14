"""
Ecotaxa Export
"""
import copy
import zipfile

import numpy as np
import pandas as pd
from skimage import img_as_ubyte
import os
import json

from morphocut.pipeline import NodeBase


def dtype_to_ecotaxa(dtype):
    try:
        if np.issubdtype(dtype, np.number):
            return "[f]"
    except TypeError:
        print(type(dtype))
        raise

    return "[t]"


class Exporter(NodeBase):
    """
    Ecotaxa Export

    Writes a .zip-Archive with the following entries:
        - ecotaxa_segmentation.csv
        - <objid>.png
        - <objid>_contours.png

    TODO: Make exported images configurable

    Input:

    {
        object_id: ...
        raw_img: {
            id: ...
            meta: {region props...},
            image: <np.array of shape = [h,w,c]>
        },
        contour_img: {
            image: <np.array of shape = [h,w,c]>
        }
    }

    """

    def __init__(self, archive_fn, img_facets, data_facets, img_ext=".jpg", loggers=[]):
        self.archive_fn = archive_fn
        self.img_facets = img_facets
        self.data_facets = data_facets
        self.img_ext = img_ext
        self.loggers = loggers

    def __call__(self, input=None):
        dataframe = []
        filenames_set = set()
        with zipfile.ZipFile(self.archive_fn, 'w', zipfile.ZIP_DEFLATED) as archive:
            for obj in input:
                # Store object data
                data = {
                    "object_id": obj["object_id"]
                }

                # Update data
                for facet_name in self.data_facets:
                    facet_data = obj["facets"][facet_name]["data"]
                    for k, v in facet_data.items():
                        data["object_{}".format(k)] = v

                # Write images
                for rank, facet_name in enumerate(self.img_facets):
                    img_data = copy.copy(data)

                    # Update data
                    img_data["img_rank"] = rank + 1
                    img_data["img_file_name"] = "{}_{}{}".format(
                        data["object_id"], facet_name, self.img_ext)

                    # Check if the filename already exists in the dataframe and stop if it does
                    # if img_data["img_file_name"] in [v['img_file_name'] for v in dataframe]:
                    #     raise ValueError('Aborting process. Object {} is already in dataframe.'.format(
                    #         img_data['img_file_name']))
                    if img_data["img_file_name"] in filenames_set:
                        raise ValueError('Aborting process. Object {} is already in dataframe.'.format(
                            img_data['img_file_name']))

                    img = img_as_ubyte(obj["facets"][facet_name]["image"])
                    _, buf = cv.imencode(self.img_ext, img)

                    archive.writestr(img_data["img_file_name"], buf.tostring())

                    dataframe.append(img_data)
                    filenames_set.add(img_data["img_file_name"])

                # Yield object for further processing
                yield obj

            # Create pandas DataFrame
            dataframe = pd.DataFrame(dataframe)

            # Drop duplicate filenames to fix bug with duplicates on ecotaxa upload
            # dataframe.drop_duplicates(subset='img_file_name', inplace=True)

            # Insert types into header
            type_header = [dtype_to_ecotaxa(
                dt) for dt in dataframe.dtypes]
            try:
                dataframe.columns = pd.MultiIndex.from_tuples(
                    list(zip(dataframe.columns, type_header)))
            except Exception as err:
                print('Exporter Exception at MultiIndex: ' + str(err))

            archive.writestr(
                "ecotaxa_export.tsv",
                dataframe.to_csv(sep='\t', encoding='utf-8', index=False))

            # export logging information
            logs = {}
            for log in self.loggers:
                logs.update(log.get_log())
            archive.writestr(
                'meta.json',
                json.dumps(logs)
            )

        print("Exported {} objects.".format(len(dataframe)))
