import ast
from pathlib import Path

import pytest
from polugins._types import ExtensionClass

from polugins_type_gen._ast import parse_from_current_env


@pytest.mark.parametrize("extension_class", ExtensionClass)
def test_generate_stubs(extension_class: ExtensionClass):
    """Tests that we can generate stubs without errors"""
    stub = ast.unparse(parse_from_current_env(extension_class))
    output_location = (Path(__file__).parent / "output" / extension_class.import_path).with_suffix(".pyi")
    output_location.parent.mkdir(exist_ok=True, parents=True)
    output_location.write_text(stub)
    ast.parse(stub)
