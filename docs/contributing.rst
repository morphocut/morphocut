Contributing
============

Open source software thrives on people's contributions.
We sincerely appreciate your interest in contributing to the MorphoCut project!

Bug Reports
-----------

Bug reports play a vital role!
Before submitting a bug report, please take a moment to check
the `GitHub issues`_ to ensure that the bug hasn't already
been reported.

.. _GitHub issues: https://github.com/morphocut/morphocut/issues

Code Contributions
------------------

Steps for Submitting Code
~~~~~~~~~~~~~~~~~~~~~~~~~

GitHub Pull Requests are the preferred method for collaborating
on code in this project.
To contribute, please follow these steps:

1. Fork the `repository`_ on GitHub.
2. Run the tests and ensure that they all pass on your system.
3. Write tests that clearly demonstrate the bug or feature you're addressing.
   Make sure these tests fail initially (a practice known as *test-driven development*).
4. Implement your changes.
5. Run the entire test suite again to verify that all tests pass,
   including the ones you just added.
6. Write `meaningful commit messages <https://chris.beams.io/posts/git-commit/>`_ to document your changes.
7. Submit a GitHub Pull Request to the main repository's ``main`` branch. Your contribution will be reviewed promptly.

.. _repository: https://github.com/morphocut/morphocut

Code Style
~~~~~~~~~~

To maintain code consistency,
please adhere to the following guidelines:

* Follow `PEP 8`_, `PEP 257`_, and the `Google Style Guide`_.
* Utilize `black <https://black.readthedocs.io/en/stable/>`_ to format your code.
* Use `isort <https://pypi.org/project/isort/>`_ to organize your imports.
* Employ `pydocstyle <https://pypi.org/project/pydocstyle/>`_ to receive feedback on your docstrings.

.. _Google Style Guide: http://google.github.io/styleguide/pyguide.html
.. _PEP 8: https://www.python.org/dev/peps/pep-0008/
.. _PEP 257: https://www.python.org/dev/peps/pep-0257/

The repository includes a ``.vscode/settings.json.default`` file that contains sensible default settings.
If you're developing in VS Code, you can use it as a starting point.

Documentation Contributions
---------------------------

Documentation holds significant value for this library,
and we warmly welcome any improvements.
The documentation resides in the ``docs/`` directory and is written in `reStructuredText`_.
We utilize `Sphinx`_ to generate a comprehensive suite of documentation,
with `napoleon`_ interpreting the docstrings.

.. _reStructuredText: http://docutils.sourceforge.net/rst.html
.. _Sphinx: http://sphinx-doc.org/index.html
.. _napoleon: https://sphinxcontrib-napoleon.readthedocs.io/en/latest/
