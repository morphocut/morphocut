Introduction
============

**MorphoCut Library** can be used to process thousands of images
almost like you would process a single image.
It was created out of the need to process large collections of images,
but is also able to treat other data types.

MorphoCut is data-type agnostic, modular, and easily parallelizable.

Writing a MorphoCut program
---------------------------

First, a :py:class:`~morphocut.core.Pipeline` is defined that
contains all operations that should be carried out on the
objects of the stream.
These operations are then applied to a whole stream of images.

MorphoCut allows concise defititions of heavily nested
image processing pipelines:

.. literalinclude:: ../examples/introduction.py
    :language: python

While creating the pipeline, everything is just placeholders.
In this step, the actions that should be performed are
*just recorded but not yet applied*.
The Nodes, therefore, don't return real values,
but identifiers for the values that will later flow through the stream.

Concepts
--------

An operation in the :ref:`Pipeline <pipelines>`
is called a ":ref:`Node <nodes>`". It usually returns one (or multiple)
:ref:`Variables <variables>`.

These are the Nodes used in this example:

.. autosummary::
    :nosignatures:

    ~morphocut.stream.Unpack
    ~morphocut.stream.Enumerate
    ~morphocut.core.Call
    ~morphocut.file.Glob
    ~morphocut.parallel.ParallelPipeline
    ~morphocut.image.ImageReader
    ~morphocut.image.FindRegions
    ~morphocut.str.Format
    ~morphocut.contrib.zooprocess.CalculateZooProcessFeatures
    ~morphocut.contrib.ecotaxa.EcotaxaWriter

.. note::
    Nodes that change the stream are labeled with "|stream|".

    :py:class:`~morphocut.stream.Unpack`, :py:class:`~morphocut.file.Glob`
    and :py:class:`~morphocut.image.FindRegions` all introduce new objects into the stream.
    Traditionally, this would be written using nested for-loops.

    MorphoCut, on the other hand, applies a sequence of processing steps
    (Nodes) which allows for easy parallelization
    and nicely decouples the individual steps in the pipeline.
