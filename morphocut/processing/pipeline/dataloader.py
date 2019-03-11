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

    def __init__(self, location, output_facet="raw", image_extensions=None):
        self.location = location
        self.output_facet = output_facet

        if image_extensions is None:
            self.image_extensions = {".jpeg", ".jpg", ".png", ".gif", ".tif"}
        else:
            self.image_extensions = set(image_extensions)

        self.files = self._find_files()

    def _find_files(self):
        """
        Scan the location for files and create index.
        """
        print("Reading location {}...".format(self.location))
        file_index = []

        for match in iglob(self.location):
            if os.path.isdir(match):
                # If the match is a path, recursively find files
                for root, dirs, files in os.walk(match):
                    rel_root = os.path.relpath(root, self.location)
                    file_index.extend(
                        os.path.join(root, f)
                        for f in files if os.path.splitext(f)[1] in self.image_extensions)

            elif os.path.isfile(match) and os.path.splitext(match)[1] in self.image_extensions:
                # If the match itself is a file, add to index
                file_index.append(match)

        return file_index

    def __call__(self, input=None):
        return _DataLoaderIter(self.files, self.output_facet)


class _DataLoaderIter:
    """
    Iterator for DataLoader that has a length.
    """

    def __init__(self, files, output_facet):
        self.files = files
        self.output_facet = output_facet

    def __len__(self):
        return len(self.files)

    def __iter__(self):
        for fn in self.files:
            print('Loading file:', fn)

            object_id = os.path.splitext(os.path.basename(fn))[0]

            try:
                image = cv.imread(fn, -1)
            except cv.error:
                print("Can't load {}.".format(fn))
                raise

            data_object = {
                "object_id": object_id,
                "facets": {
                    self.output_facet: dict(
                        image=image
                    )
                }
            }

            yield data_object
