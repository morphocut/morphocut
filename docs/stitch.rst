Stitching
=========

A :py:class:`~morphocut.stitch.Stitch` node allows to combine multiple regions into one larger frame
while providing the same interface as a Numpy array.
In contrast to a regular Numpy array, :py:class:`morphocut.stitch.Frame` is a sparse format
that only stores regions with content.

.. automodule:: morphocut.stitch
    :members: