Core API
========

.. module:: morphocut.core

This chapter describes the core module of MorphoCut.

Pipelines
---------

The Pipeline is the main entry point for a MorphoCut application.

.. autoclass:: Pipeline
    :members:

Nodes
-----

A Node applies creates, updates or deletes stream objects.

LambdaNode
~~~~~~~~~~

In simple cases, a regular function
can be converted into a pipeline node using :py:class:`LambdaNode`.

.. autoclass:: LambdaNode
    :members:

Subclassing :py:class:`Node`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If a Node has multiple outputs or needs to change the stream,
:py:class:`Node` has to be subclassed.

The subclass has no or any number of :py:obj:`@Output <Output>` decorators.
:py:obj:`@ReturnOutputs <ReturnOutputs>` is used to turn :py:class:`Node`
subclasses into a functions returning stream variables.

Overriding :py:meth:`transform <Node.transform>`
''''''''''''''''''''''''''''''''''''''''''''''''

If the Node handles one object at a time,
it is enough to implement :py:meth:`Node.transform`.
:py:meth:`Node.transform_stream` will use introspection
to call it with the right parameters.

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

.. autodecorator:: Output(name, type, doc)

.. autodecorator:: ReturnOutputs

Variables
---------

Upon instanciation, Nodes return Variables
that are used to identify values in stream objects.

.. autoclass:: Variable
    :members:
