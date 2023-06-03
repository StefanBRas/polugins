from enum import Enum
from pathlib import Path
from typing import Callable
import polars as pl

class Namespace(Enum):
    """ TODO: rename this. Horrible naming. """

    EXPR = "expr"
    SERIES = "series"
    LAZYFRAME = "lazyframe"
    DATAFRAME = "dataframe"

    def to_reg_function(self) -> Callable:
        if self == Namespace.EXPR:
            return pl.api.register_expr_namespace
        if self == Namespace.SERIES:
            return pl.api.register_series_namespace
        if self == Namespace.LAZYFRAME:
            return pl.api.register_lazyframe_namespace
        if self == Namespace.DATAFRAME:
            return pl.api.register_dataframe_namespace
        raise TypeError  # will never happen

    @property
    def import_path(self) -> Path:
        if self == Namespace.EXPR:
            return Path("polars", "expr", "expr")
        if self == Namespace.SERIES:
            return Path("polars", "series", "series")
        if self == Namespace.LAZYFRAME:
            return Path("polars", "lazyframe", "frame")
        if self == Namespace.DATAFRAME:
            return Path("polars", "dataframe", "frame")
        raise TypeError  # will never happen


    @classmethod
    def from_entrypoint_group(cls, group: str):
        namespace_type = group.split(".")[1]
        return cls(namespace_type)

