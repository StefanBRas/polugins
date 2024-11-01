import importlib
from enum import Enum
from pathlib import Path
from typing import Callable

import polars as pl
from typing_extensions import Self


class ExtensionClass(Enum):
    EXPR = "expr"
    SERIES = "series"
    LAZYFRAME = "lazyframe"
    DATAFRAME = "dataframe"

    def to_reg_function(self) -> Callable:
        if self == ExtensionClass.EXPR:
            return pl.api.register_expr_namespace
        if self == ExtensionClass.SERIES:
            return pl.api.register_series_namespace
        if self == ExtensionClass.LAZYFRAME:
            return pl.api.register_lazyframe_namespace
        if self == ExtensionClass.DATAFRAME:
            return pl.api.register_dataframe_namespace
        raise TypeError  # will never happen

    def register(self, name: str, namespace: str):
        module, namespace_class = namespace.split(":")
        namespace_module = importlib.import_module(module)
        namespace = getattr(namespace_module, namespace_class)
        if self == ExtensionClass.EXPR:
            if name in pl.Expr._accessors and getattr(pl.Expr, name) is namespace:
                return
            return pl.api.register_expr_namespace(name)(namespace)  # type: ignore
        if self == ExtensionClass.SERIES:
            if name in pl.Series._accessors and getattr(pl.Series, name) is namespace:
                return
            return pl.api.register_series_namespace(name)(namespace)  # type: ignore
        if self == ExtensionClass.LAZYFRAME:
            if name in pl.LazyFrame._accessors and getattr(pl.LazyFrame, name) is namespace:
                return
            return pl.api.register_lazyframe_namespace(name)(namespace)  # type: ignore
        if self == ExtensionClass.DATAFRAME:
            if name in pl.DataFrame._accessors and getattr(pl.DataFrame, name) is namespace:
                return
            return pl.api.register_dataframe_namespace(name)(namespace)  # type: ignore
        raise TypeError  # will never happen, poor mans pattern matching.

    @property
    def import_path(self) -> Path:
        if self == ExtensionClass.EXPR:
            return Path("polars", "expr", "expr")
        if self == ExtensionClass.SERIES:
            return Path("polars", "series", "series")
        if self == ExtensionClass.LAZYFRAME:
            return Path("polars", "lazyframe", "frame")
        if self == ExtensionClass.DATAFRAME:
            return Path("polars", "dataframe", "frame")
        raise TypeError  # will never happen

    @classmethod
    def from_entrypoint_group(cls, group: str) -> Self:
        namespace_type = group.split(".")[1]
        return cls(namespace_type)
