import pytest
from contextlib import contextmanager
from typing import TYPE_CHECKING

from polars.dataframe import DataFrame
from polars.expr import Expr
from polars.lazyframe import LazyFrame
from polars.series import Series


@pytest.fixture(autouse=True)
def _fresh_accessors():
    classes = [DataFrame, Expr, LazyFrame, Series]
    accessors_before = [cls._accessors.copy() for cls in classes]

    yield

    accessors_after = [cls._accessors.copy() for cls in classes]
    for before, after, cls in zip(accessors_before, accessors_after, classes):
        all_added = after - before
        for name in all_added:
            cls._accessors.remove(name)
            delattr(cls, name)


