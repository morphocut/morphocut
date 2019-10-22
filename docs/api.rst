Core API
========

.. module:: morphocut.core

Pipeline
--------

The Pipeline is the main entry point for a MorphoCut application.

.. autoclass:: Pipeline
    :members:

Node
----

.. autoclass:: Node
    :members:

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

.. TODO: Simple and complex

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
                return ham, spam

.. autodecorator:: Output(name, type, doc)

.. autodecorator:: ReturnOutputs
