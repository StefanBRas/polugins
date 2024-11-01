import contextlib
import os

import pytest
from polars import LazyFrame

from polugins.main import register_namespaces


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


@contextlib.contextmanager
def set_env(**environ):
    old_environ = dict(os.environ)
    os.environ.update(environ)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(old_environ)


def test_suppress_warning(recwarn: pytest.WarningsRecorder):
    with set_env(
        polugins_lazyframe_selfregister="polugins._testing.self_register:SelfRegisterNamespace"
    ):
        register_namespaces(load_env=True, load_config=False, load_entrypoints=False)
        ldf = LazyFrame()
        ldf.selfregister.custom_method(x=1)
    assert len(recwarn) == 0
