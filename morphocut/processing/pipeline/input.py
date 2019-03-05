"""
Input nodes
"""

import os
from morphocut.processing.pipeline import NodeBase


class LocalDirectoryInput(NodeBase):
    """
    Read the contents of a local directory and yield processing objects.

    Source node (no predecessors).
    """

    def __init__(self, location, object_extensions=None):
        self.location = location
        self._index = None
        self.object_extensions = object_extensions

    def get_options(self):
        index = self._get_index()

        if self.index is None:
            self._make_index()

        extensions = set(os.path.splitext(f)[1] for f in self.index["files"])

        object_extensions = {".jpeg", ".jpg", ".png", ".gif", ".tif"}

        object_extensions &= extensions

        index_extensions = {".tsv", ".csv"}

        index_files = [f for f in self.index["files"]
                       if os.path.splitext(f)[1] in index_extensions]

        return {
            "object_extensions": object_extensions,
            "index_files": index_files,
        }

    def _get_index(self):
        """
        Scan the location for files and create index.
        """
        print("Reading location {}...".format(self.location))
        index = {"dirs": [], "files": [], "root_files": []}
        for root, dirs, files in os.walk(os.path.abspath(self.location)):
            rel_root = os.path.relpath(root, self.location)
            print(rel_root)
            index["dirs"].extend(os.path.join(rel_root, d) for d in dirs)
            index["files"].extend(os.path.join(rel_root, f) for f in files)
            index["root_files"].extend(os.path.join(root, f) for f in files)
        self.index = index

    def __call__(self, input=None):
        i = 0
        while i < len(self.index['root_files']):
            yield self.index['root_files'][i]
            i += 1
