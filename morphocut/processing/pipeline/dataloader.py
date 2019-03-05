import os

import cv2 as cv
from morphocut.processing.pipeline import NodeBase
from glob import iglob


class DataLoader(NodeBase):
    """
    Read the contents of a local directory and yield processing objects.

    Source node (no predecessors).

    Output:

    {
        object_id: ...
        facets: {
            input_data: {
                meta: {filename: ...},
                image: <np.array of shape = [h,w,c]>
            }
        }
    }
    """

    def __init__(self, location, object_extensions=None):
        self.location = location
        self._index = None
        self.object_extensions = object_extensions

        self._get_index()

    def get_options(self):
        self._get_index()

        if self.index is None:
            self._make_index()

        extensions = set(os.path.splitext(f['filepath'])[-1]
                         for f in self.index["files"])

        object_extensions = {".jpeg", ".jpg", ".png", ".gif", ".tif"}

        object_extensions &= extensions

        index_extensions = {".tsv", ".csv"}

        index_files = [f for f in self.index["files"]
                       if os.path.splitext(f['filepath'])[-1] in index_extensions]

        return {
            "object_extensions": object_extensions,
            "index_files": index_files,
        }

    def _get_index(self):
        """
        Scan the location for files and create index.
        """
        print("Reading location {}...".format(self.location))
        self.object_extensions = {".jpeg", ".jpg", ".png", ".gif", ".tif"}
        index = {"dirs": [], "files": [], "root_files": []}
        for match in iglob(os.path.abspath(self.location)):
            for root, dirs, files in os.walk(match):
                rel_root = os.path.relpath(root, self.location)
                index["dirs"].extend(os.path.join(rel_root, d) for d in dirs)
                index["files"].extend(dict(
                    filename=f.split('.')[0],
                    filepath=os.path.join(root, f)
                ) for f in files if os.path.splitext(f)[1] in self.object_extensions)
        self.index = index

    def __call__(self, input=None):
        return DataLoaderIter(self.index["files"])


class DataLoaderIter:
    def __init__(self, files):
        self.files = files

    def __len__(self):
        return len(self.files)

    def __iter__(self):
        for file in self.files:
            print('Loading file ' + file['filepath'])
            data_object = dict(
                object_id=file['filename'],
                facets=dict(
                    input_data=dict(
                        meta=dict(
                            filename=file['filename'],
                            filepath=file['filepath'],
                        ),
                        image=cv.imread(file['filepath'])
                    )
                ))
            yield data_object
