Changelog
=========

0.2.x
-----

Added
~~~~~

- Added `batch.BatchPipeline`: Combine consecutive objects into a batch.

- Added `mjpeg_streamer.MJPEGStreamer`: Stream images via HTTP (e.g. to the Browser). (#75)

- Added `integration.raspi.PiCameraReader`: Read frames from the Raspberry Pi's camera. (#75)

- Added `filters` as a replacement for `stat`. (#77)

- Added `StreamEstimator` to estimate the remaining length of a stream. (#79)

- Added `utils.stream_groupby`: Split a stream into sub-streams by key.

- Added support for Python 3.9 and 3.10 (#87).

- Added `StreamObject.child`.

- Added `HDF5Writer`.


Changed
~~~~~~~

- Use `UnavailableObject` instead of `import_optional_dependency`.

- Make `pandas` and `tqdm` a hard dependency.

- Require scikit-image>=0.19 (#86)

- Implemented `n_remaining_hint` in file.


Deprecated
~~~~~~~~~~

- `stream.TQDM` in favor of `stream.Progress`
- Deprecate `stat` (#77).

Removed
~~~~~~~

- Drop `import_optional_dependency` copied over from pandas.

- Dropped support for Python 3.6 (#87).

Fixed
~~~~~

- ValueError: 'version' argument is required in Sphinx directives #80
- UnknownArchiveError: Close EcoTaxa archives (#88)
