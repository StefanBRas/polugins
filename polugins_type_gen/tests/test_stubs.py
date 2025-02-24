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


def test_default_positional_args():
    """Assert that we keep default args. Issue 88"""
    extension_class = ExtensionClass.DATAFRAME
    parsed_ast = parse_from_current_env(extension_class)
    class_defs = [element for element in parsed_ast.body if isinstance(element, ast.ClassDef)]
    dataframe_node = next(node for node in class_defs if node.name == "DataFrame")
    df_methods = [element for element in dataframe_node.body if isinstance(element, ast.FunctionDef)]
    df_init = next(node for node in df_methods if node.name == "__init__")
    assert df_init.args.defaults
