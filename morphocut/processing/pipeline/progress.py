from tqdm import tqdm

from morphocut.processing.pipeline import NodeBase


class Progress(NodeBase):
    def __init__(self, desc=None):
        self.desc = desc

    def __call__(self, input=None):
        return tqdm(input, desc=self.desc)
