The MorphoCut image processing pipeline library
===============================================

.. image:: _static/morphocut_logo.png
    :align: right
    :width: 50%

MorphoCut is a pipelined image processing library designed to handle
large volumes of biological, medical, oceanographic and remote sensing
image data. It is modular from ground up and comes with "`batteries
included`_" to provide common processing steps like segmentation and
feature extraction. It reads and writes many :doc:`formats <formats>`
like video,
`Bio-Formats <https://docs.openmicroscopy.org/bio-formats/latest/supported-formats.html>`__
and `EcoTaxa <https://ecotaxa.obs-vlfr.fr/>`__.

.. _`batteries included`: https://en.wikipedia.org/wiki/Batteries_Included

User Guide
----------

.. toctree::
   :maxdepth: 2

   introduction
   why-not-xy
   installation
   image
   str
   file
   stream
   batch
   filters
   stat
   scalebar
   parallel
   hdf5
   formats
   profiling
   core
   contrib
   torch
   integration
   mjpeg_streamer
   utils
   examples
   contributing
   authors

Acknowledgements
----------------

This project was made possible by the support from
`SFB754 - Climate Biogeochemistry Interactions in the Tropical Ocean <https://www.sfb754.de/>`__.
