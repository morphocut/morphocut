Introduction
============

**MorphoCut Library** can be used to process thousands of images
almost like you would process a single image.

It was created out of the need to process large collections of images,
but is also able to treat other data types.

First, a :py:class:`~morphocut.core.Pipeline` is defined that
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

While creating the pipeline, everything is just placeholders.
In this step, the actions that should be performed are
*just recorded but not yet applied*.
The Nodes, therefore, don't return real values,
but identifiers for the values that will later flow through the stream.

Concepts
--------

- :py:class:`~morphocut.core.Call`: Record the call to a regular function.

Another
    Foo bar.
    blabl
