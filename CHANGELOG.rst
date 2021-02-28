Changelog
=========

0.2.x
-----

Added
~~~~~

- Added `mjpeg_streamer.MJPEGStreamer`: Stream images via HTTP (e.g. to the Browser). (#75)
- Added `integration.raspi.PiCameraReader`: Read frames from the Raspberry Pi's camera. (#75)
- Added `filters` as a replacement for `stat` (#77).

Changed
~~~~~~~

- Use `UnavailableObject` instead of `import_optional_dependency`.
- Make `pandas` and `tqdm` a hard dependency.

Deprecated
~~~~~~~~~~

- Deprecate `stat` (#77).

Removed
~~~~~~~

- Drop `import_optional_dependency` copied over from pandas.

Fixed
~~~~~
