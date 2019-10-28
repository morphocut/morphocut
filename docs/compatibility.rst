Compatibility
=============

MorphoCut is compatible with Python 3.6 and 3.7.

.. note::

    As of today (Oct 2019) it is not fully compatible with Pylint (2.3.1)
    and Mypy (0.720) because they both don't detect type changes by
    decorators:

    - Pylint: `Type changes in decorators are not detected <https://github.com/PyCQA/pylint/issues/2578>`_
    - Mypy: `Support function decorators excellently <https://github.com/python/mypy/issues/3157>`_
