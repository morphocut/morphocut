The MorphoCut image processing pipeline library
===============================================

.. image:: _static/morphocut_logo.png
    :align: right
    :width: 50%

MorphoCut is a pipelined image processing library designed to handle
large volumes of biological, medical, oceanographic and remote sensing
image data. It is modular from ground up and comes with "batteries
included" to provide common processing steps like segmentation and
feature extraction. It reads and writes many :doc:`formats <formats>`
like video,
`Bio-Formats <https://docs.openmicroscopy.org/bio-formats/latest/supported-formats.html>`__
and `EcoTaxa <https://ecotaxa.obs-vlfr.fr/>`__.

The MorphoCut library is accompanied by a `web
application <https://github.com/morphocut/morphocut-server>`__ that
serves as a front end to the library.

User Guide
----------

.. toctree::
   :maxdepth: 2

   installation
   quickstart
   formats
   image
   str
   file
   stream
   parallel
   image
   api
   contributing
   authors

Acknowledgements
----------------

This project was made possible by the support from
`SFB754 - Climate Biogeochemistry Interactions in the Tropical Ocean <https://www.sfb754.de/>`__.
