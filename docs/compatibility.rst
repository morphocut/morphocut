Compatibility
=============

MorphoCut is compatible with Python 3.6 and 3.7.

.. note::

    As of today (Oct 2019) it is not fully compatible with Pylint (2.3.1)
    and Mypy (0.720) because they both don't detect type changes by
    decorators:

    - Pylint: |pylint-issue| `Type changes in decorators are not detected <https://github.com/PyCQA/pylint/issues/2578>`_
    - Mypy: |mypy-issue| `Support function decorators excellently <https://github.com/python/mypy/issues/3157>`_


.. |pylint-issue| image:: https://img.shields.io/github/issues/detail/state/PyCQA/pylint/2578
    :target: https://github.com/PyCQA/pylint/issues/2578

.. |mypy-issue| image:: https://img.shields.io/github/issues/detail/state/python/mypy/3157
    :target: https://github.com/python/mypy/issues/3157
