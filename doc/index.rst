.. MorphoCut documentation master file, created by
   sphinx-quickstart on Fri Sep 20 20:41:31 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to MorphoCut's documentation!
=====================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

Installation
------------

**MorphoCut** is packaged on PyPI and can be installed with pip:

.. code-block:: sh

    pip install morphocut

Be warned that this package is currently under heavy development
and anything might change any time!

To install the development version, do:

.. code-block:: sh

    pip install -U git+https://github.com/morphocut/morphocut.git

Getting started
---------------

Use **MorphoCut** like you would process a single image.
The operations are then automatically applied to a whole stream of images.

.. literalinclude:: ../examples/simple.py
    :language: python

Further reading
---------------

* :doc:`input_formats`

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
