Core API
========

.. module:: morphocut.core

This chapter describes the core module of MorphoCut.

.. _pipelines:

Pipelines
---------

The Pipeline is the main entry point for a MorphoCut application.

.. autoclass:: Pipeline
    :members:

.. _nodes:

Nodes
-----

A Node applies creates, updates or deletes stream objects.

Call
~~~~~~~~~~

In simple cases, a call to a regular function
can be recorded in a pipeline using :py:class:`Call`.

.. autoclass:: Call
    :members:

Subclassing :py:class:`Node`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If a Node has multiple outputs or needs to change the stream,
:py:class:`Node` has to be subclassed.

The subclass has no or any number of :py:obj:`@Output <Output>` decorators.
:py:obj:`@ReturnOutputs <ReturnOutputs>` is used to turn :py:class:`Node`
subclasses into a functions returning stream variables.

Overriding :code:`transform(...)`
'''''''''''''''''''''''''''''''''

If the Node handles one object at a time,
it is enough to implement a custom :code:`transform(...)`.

The parameter names have to correspond to attributes of the Node.
:py:meth:`Node.transform_stream` will then use introspection
to call :code:`transform` with the right parameter values.

.. code-block:: python

    @ReturnOutputs
    @Output("bar")
    @Output("baz")
    class Foo(Node):
        """This class has two outputs."""
        def __init__(self, ham, spam):
            super().__init__()
            self.ham = ham
            self.spam = spam

        # This is automatically called by Node.transform_stream,
        # reading ham and spam from the stream
        # and introducing the result back into the stream.
        def transform(self, ham, spam):
            # ham and spam are raw values here
            return ham + spam, ham - spam

    with Pipeline() as pipeline:
        # ham and spam are stream variables here
        ham = ...
        spam = ...
        bar, baz = Foo(ham, spam)

Overriding :py:meth:`transform_stream <Node.transform_stream>`
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

If the Node has to change the stream in some way,
:py:meth:`Node.transform_stream` needs to be overridden.

.. code-block:: python

    @ReturnOutputs
    @Output("bar")
    @Output("baz")
    class Foo(Node):
        """This class has two outputs."""
        def __init__(self, ham, spam):
            super().__init__()
            self.ham = ham
            self.spam = spam

        def transform_stream(self, stream):
            with closing_if_closable(stream):
                for obj in stream:
                    # Retrieve raw values
                    ham, spam = self.prepare_input(obj, ("ham", "spam"))

                    # Remove objects from stream based on some condition
                    if not ham:
                        continue

                    # Append new values to the stream object:
                    # bar = ham + spam
                    # baz = ham - spam
                    yield self.prepare_output(obj, ham + spam, ham - spam)

    with Pipeline() as pipeline:
        # ham and spam are stream variables here
        ham = ...
        spam = ...
        bar, baz = Foo(ham, spam)

.. autoclass:: Node
    :members:

.. autodecorator:: Output

.. autodecorator:: ReturnOutputs

.. _variables:

Variables
---------

Upon instanciation, Nodes return Variables
that are used to identify values in stream objects.

.. autoclass:: Variable

Stream
------

.. autodata:: Stream
    :annotation:

.. autoclass:: StreamObject
    :members:
