"""
Custom mypy plugin to solve the temporary problem with untyped decorators.

This problem appears when we try to change the return type of the function.
However, currently it is impossible due to this bug:
https://github.com/python/mypy/issues/3157

This plugin is a temporary solution to the problem.
It should be later replaced with the official way of doing things.

``mypy`` API docs are here:
https://mypy.readthedocs.io/en/latest/extending_mypy.html

Adapted from https://github.com/dry-python/returns: contrib/mypy/decorator_plugin.py

Copyright 2016-2019 dry-python organization

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

1. Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the
  distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

from typing import Callable, Optional, Type

from mypy.plugin import FunctionContext, Plugin
from mypy.types import CallableType

#: Set of full names of our decorators.
_TYPED_DECORATORS = {
    'morphocut.core.ReturnOutputs',
    'morphocut.core.Output',
}


def _change_decorator_function_type(
    decorated: CallableType,
    decorator: CallableType,
) -> CallableType:
    """Replaces revealed argument types by mypy with types from decorated."""
    decorator.arg_types = decorated.arg_types
    decorator.arg_kinds = decorated.arg_kinds
    decorator.arg_names = decorated.arg_names
    return decorator


def _analyze_decorator(function_ctx: FunctionContext):
    """Tells us what to do when one of the typed decorators is called."""
    if not isinstance(function_ctx.arg_types[0][0], CallableType):
        print(function_ctx, "default_return_type")
        return function_ctx.default_return_type
    if not isinstance(function_ctx.default_return_type, CallableType):
        print(function_ctx, "default_return_type")
        return function_ctx.default_return_type

    print(function_ctx, "_change_decorator_function_type")
    return _change_decorator_function_type(
        function_ctx.arg_types[0][0],
        function_ctx.default_return_type,
    )


class _TypedDecoratorPlugin(Plugin):
    def get_function_hook(  # type: ignore
        self, fullname: str,
    ) -> Optional[Callable[[FunctionContext], Type]]:
        """
        One of the specified ``mypy`` callbacks.

        Runs on each function call in the source code.
        We are only interested in a particular subset of all functions.
        So, we return a function handler for them.

        Otherwise, we return ``None``.
        """
        if fullname in _TYPED_DECORATORS:
            return _analyze_decorator
        return None


def plugin(version: str) -> Type[Plugin]:
    """Plugin's public API and entrypoint."""
    return _TypedDecoratorPlugin
