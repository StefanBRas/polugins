from pathlib import Path

import pytest
from polugins._types import ExtensionClass

from polugins_type_gen.cli import (
    NoNamespaceRegisteredException,
    create_stubs,
)

typings_dir = Path("typings")


@pytest.mark.parametrize(
    "venv_path",
    [
        None,
        pytest.param(
            Path(".venv"),
            marks=[pytest.mark.skip(reason="Fails in CI. Should fix after changing to uv.")],
        ),
    ],
)
def test_cli(venv_path: Path):
    create_stubs(venv_path)
    lazyframe_stubs_path = (typings_dir / ExtensionClass.LAZYFRAME.import_path).with_suffix(".pyi")
    lazyframe_stubs = lazyframe_stubs_path.read_text()
    assert "external: example_package.PackageNamespace" in lazyframe_stubs
    assert "external_nested: example_package.some_module.some_file.NestedPackageNamespace"
    assert "external_nested_init: example_package.some_module_from_init.NestedInitPackageNamespace"
    assert "external_nested_same_module_1: example_package.same_module.some_file_1.NestedSameModule1PackageNamespace"
    assert "external_nested_same_module_2: example_package.same_module.some_file_2.NestedSameModule2PackageNamespace"
    assert "pyproject: polugins._testing.namespaces.PyProjectNameSpace" in lazyframe_stubs
    assert "config: polugins._testing.namespaces.ConfigNameSpace" in lazyframe_stubs
    assert "env: polugins._testing.namespaces.EnvNameSpace" in lazyframe_stubs

    assert "import example_package" in lazyframe_stubs
    assert "import example_package.some_module.some_file"
    assert "import example_package.some_module_from_init"
    assert "import example_package.same_module.some_file_1"
    assert "import example_package.same_module.some_file_2"
    assert "import polugins._testing.namespaces" in lazyframe_stubs
    assert "import polugins._testing.namespaces" in lazyframe_stubs
    assert "import polugins._testing.namespaces" in lazyframe_stubs


def test_exception_on_no_namespaces(mocker):
    mocker.patch(
        "polugins_type_gen.cli._get_namespaces",
        return_value={
            ExtensionClass.EXPR: {},
            ExtensionClass.SERIES: {},
            ExtensionClass.LAZYFRAME: {},
            ExtensionClass.DATAFRAME: {},
        },
    )
    with pytest.raises(NoNamespaceRegisteredException):
        create_stubs()
