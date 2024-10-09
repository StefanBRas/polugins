import ast
from pathlib import Path

import pytest
from polugins._types import ExtensionClass

from polugins_type_gen._ast import parse_from_current_env


stubs_dir = Path(__file__).parent.parent / Path("src", "polugins_type_gen", "_stubs")
all_stub_files = sorted(p for p in stubs_dir.rglob("*") if p.is_file())


@pytest.mark.parametrize("extension_class", ExtensionClass)
def test_generate_stubs(extension_class: ExtensionClass):
    """ Tests that we can generate stubs without errors """
    stub = parse_from_current_env(extension_class)
    ast.parse(ast.unparse(stub))
