Changelog
=========

0.2.x
-----

Added
~~~~~

- Added `torch.PyTorch`: Apply a PyTorch module. (#95)

- Added `batch.BatchPipeline`: Combine consecutive objects into a batch. (#92)

- Added `mjpeg_streamer.MJPEGStreamer`: Stream images via HTTP (e.g. to the Browser). (#75)

- Added `integration.raspi.PiCameraReader`: Read frames from the Raspberry Pi's camera. (#75)

- Added `filters` as a replacement for `stat`. (#77)

- Added `utils.StreamEstimator` to estimate the remaining length of a stream. (#79, #91)

- Added `utils.stream_groupby`: Split a stream into sub-streams by key.

- Added support for Python 3.9 and 3.10 (#87).

Changed
~~~~~~~

- EcotaxaReader: Return EcotaxaObject. (#102)

- EcotaxaWriter: Allow Variables for `archive_fn`. (#100)

- Use `UnavailableObject` instead of `import_optional_dependency`.

- Make `pandas` and `tqdm` a hard dependency.

- Require scikit-image>=0.19 (#86)

Deprecated
~~~~~~~~~~

- Deprecate `stat` (#77).

Removed
~~~~~~~

- Drop `import_optional_dependency` copied over from pandas.

- Dropped support for Python 3.6 (#87).

Fixed
~~~~~

- calculation of `n_remaining_hint` in `stream.Slice` (#111). 

- ValueError: 'version' argument is required in Sphinx directives (#80).

- UnknownArchiveError: Close EcoTaxa archives (#88).

- wrongly reported n_remaining_hint in Progress after Slice (#105).
