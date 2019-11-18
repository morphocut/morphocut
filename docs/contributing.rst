Contributing
============

Open source software thrives on people's contributions.
Thank you for considering contributing to the MorphoCut project!

Bug Reports
-----------

Bug reports are very important!
Have a look at the `GitHub issues`_ to confirm that the bug
hasn't been reported before.

.. _GitHub issues: https://github.com/morphocut/morphocut/issues

Code Contributions
------------------

Steps for Submitting Code
~~~~~~~~~~~~~~~~~~~~~~~~~

GitHub Pull Requests are the expected method of code collaboration on this
project.

Please follow these  steps:

1. Fork the `repository`_ on GitHub.
2. Run the tests and make sure that they all pass on your system.
3. Write tests that demonstrate your bug or feature. Ensure that they fail.
   (This is called *test-driven development*.)
4. Make your change.
5. Run the entire test suite again, ensuring that all tests pass *including
   the ones you just added*.
6. Write `meaningful commit messages <https://chris.beams.io/posts/git-commit/>`_.
7. Send a GitHub Pull Request to the main repository's ``master`` branch.
   Your contribution will then be reviewed.

.. _repository: https://github.com/morphocut/morphocut

Code style
~~~~~~~~~~

* Follow `PEP 8`_, `PEP 257`_ and the `Google Style Guide`_.
* Use `black <https://black.readthedocs.io/en/stable/>`_ to format your code.
* Use `isort <https://pypi.org/project/isort/>`_ to sort your imports.
* Use `pydocstyle <https://pypi.org/project/pydocstyle/>`_ to get feedback on your docstrings.

.. _Google Style Guide: http://google.github.io/styleguide/pyguide.html
.. _PEP 8: https://www.python.org/dev/peps/pep-0008/
.. _PEP 257: https://www.python.org/dev/peps/pep-0257/

The repository contains a ``.vscode/settings.json.default`` file that contains
the required settings.
Use it as a starting point if you're developing in VS Code.


Documentation Contributions
---------------------------

Documentation is a very important part of this library
and improvements are very welcome!
It lives in the ``docs/`` directory and is written in
`reStructuredText`_. We use `Sphinx`_ to generate the full suite of
documentation. `napoleon`_ is used to interpret the docstrings.

.. _reStructuredText: http://docutils.sourceforge.net/rst.html
.. _Sphinx: http://sphinx-doc.org/index.html
.. _napoleon: https://sphinxcontrib-napoleon.readthedocs.io/en/latest/
