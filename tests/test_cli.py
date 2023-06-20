from pathlib import Path
from polugins.cli import create_stubs
from polugins._types import ExtensionClass
import shutil
import pytest
from polars import __version__ as pl_version


typings_dir = Path("typings")

@pytest.fixture
def typings_directory():
    if typings_dir.exists():
        shutil.rmtree(typings_dir)
    yield
    if typings_dir.exists():
        shutil.rmtree(typings_dir)

def test_cli(typings_directory):
    create_stubs(pl_version)
    lazyframe_stubs_path = (typings_dir / ExtensionClass.LAZYFRAME.import_path).with_suffix(".pyi")
    lazyframe_stubs = lazyframe_stubs_path.read_text()
    assert 'external: example_package.PackageNamespace' in lazyframe_stubs
    assert 'pyproject: polugins._testing.namespaces.PyProjectNameSpace' in lazyframe_stubs
    assert 'config: polugins._testing.namespaces.ConfigNameSpace' in lazyframe_stubs
    assert 'env: polugins._testing.namespaces.EnvNameSpace' in lazyframe_stubs

