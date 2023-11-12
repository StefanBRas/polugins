import pytest
from polars import LazyFrame

from polugins.main import register_namespaces


def test_custom():
    ldf = LazyFrame()
    from polugins._testing.namespaces import CustomNamespace

    register_namespaces(
        lazyframe_namespaces={
            "custom": CustomNamespace,
            "custom_path": "polugins._testing.namespaces:CustomNamespaceByPath",
        },
        load_entrypoints=False,
        load_config=False,
        load_env=False,
    )
    ldf.custom.custom_method(x=1)
    ldf.custom_path.custom_method(x=1)


def test_external():
    ldf = LazyFrame()
    register_namespaces(load_entrypoints=True, load_config=True, load_env=True)
    ldf.external.some_method(x=1)
    ldf.pyproject.custom_method(x=1)
    ldf.config.custom_method(x=1)
    ldf.env.custom_method(x=1)


def test_pl_reexport():
    from polugins import pl

    ldf = pl.LazyFrame()
    ldf.external.some_method(x=1)
    ldf.pyproject.custom_method(x=1)
    ldf.config.custom_method(x=1)


def test_fresh_accessors():
    """Test that conftest.fresh_accessors correctly deletes added accessors"""
    ldf = LazyFrame()
    with pytest.raises(AttributeError):
        ldf.config  # type: ignore
    with pytest.raises(AttributeError):
        ldf.custom  # type: ignore
