import ast

import pytest
from polugins._types import ExtensionClass

from polugins_type_gen._ast import parse_from_current_env


@pytest.mark.parametrize("extension_class", ExtensionClass)
def test_generate_stubs(extension_class: ExtensionClass):
    """Tests that we can generate stubs without errors"""
    stub = parse_from_current_env(extension_class)
    ast.parse(ast.unparse(stub))
