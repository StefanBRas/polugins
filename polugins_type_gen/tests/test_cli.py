import shutil
from pathlib import Path

import pytest
from polugins._types import ExtensionClass

from polugins_type_gen.cli import (
    MissingVersionException,
    NoNamespaceRegisteredException,
    create_stubs,
)

typings_dir = Path("typings")


@pytest.fixture
def typings_directory():
    if typings_dir.exists():
        shutil.rmtree(typings_dir)
    yield
    if typings_dir.exists():
        shutil.rmtree(typings_dir)


def test_cli(typings_directory):
    create_stubs("0.20.3")
    lazyframe_stubs_path = (typings_dir / ExtensionClass.LAZYFRAME.import_path).with_suffix(".pyi")
    lazyframe_stubs = lazyframe_stubs_path.read_text()
    assert "external: example_package.PackageNamespace" in lazyframe_stubs
    assert "pyproject: polugins._testing.namespaces.PyProjectNameSpace" in lazyframe_stubs
    assert "config: polugins._testing.namespaces.ConfigNameSpace" in lazyframe_stubs
    assert "env: polugins._testing.namespaces.EnvNameSpace" in lazyframe_stubs


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
        create_stubs("0.20.3")


def test_exception_on_non_existing_version():
    with pytest.raises(MissingVersionException):
        create_stubs("9999.99.9")
