Introduction
============

**MorphoCut Library** can be used to process thousands of images
almost like you would process a single image.

It was created out of the need to process large collections of images,
but is also able to treat other data types.

First, a py:class:`~morphocluster.core.Pipeline` is defined that
contains all operations.
These operations are then automatically applied to a whole stream of images.

Selling points
--------------
- Modular
- Parallelizable

MorphoCut allows concise defititions of heavily nested
image processing pipelines:

.. literalinclude:: ../examples/quickstart.py
    :language: python

Concepts
--------

- :py:class:`LambdaNode <morphocut.core.LambdaNode>`:
  Foo bar.
  blabla.
  blabla.

Another
    Foo bar.
    blabl
