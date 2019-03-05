import os

import cv2 as cv
from morphocut.processing.pipeline import NodeBase
from glob import iglob


class DatabasePersistor(NodeBase):
    """
    Creates new objects and stores them in the database.
    """

    def __init__(self):
        # TODO
