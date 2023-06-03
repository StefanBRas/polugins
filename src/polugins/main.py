from typing import Callable, Dict, Optional, Type
import polars as pl
import sys
from enum import Enum, auto


# TODO: Stolen from pluggy, remember license and stuff
if sys.version_info >= (3, 8):
    from importlib import metadata as importlib_metadata
else:
    # TODO: remember that this is only needed for python < 3.8
    import importlib_metadata  # ignore


def load_setuptools_entrypoints():
    """
    TODO: Stolen from pluggy, remember license and stuff
    """
    namespaces = {
        Namespace.EXPR: {},
        Namespace.SERIES: {},
        Namespace.LAZYFRAME: {},
        Namespace.DATAFRAME: {},
    }
    for dist in list(importlib_metadata.distributions()):
        for ep in dist.entry_points:
            if ep.group.startswith("polugins"):
                namespace = ep.load()
                namespaces[Namespace.from_entrypoint_group(ep.group)][ep.name] = namespace
    return namespaces


class Namespace(Enum):
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

    @classmethod
    def from_entrypoint_group(cls, group: str):
        namespace_type = group.split(".")[1]
        return cls(namespace_type)



def register_namespaces(
    lazyframe_namespaces: Dict[str, Type] = {},
    dataframe_namespaces: Dict[str, Type] = {},
    expression_namespaces: Dict[str, Type] = {},
    series_namespaces: Dict[str, Type] = {},
    entrypoints: bool = True,
):
    for name, namespace in lazyframe_namespaces.items():
        pl.api.register_lazyframe_namespace(name)(namespace)
    for name, namespace in dataframe_namespaces.items():
        pl.api.register_dataframe_namespace(name)(namespace)
    for name, namespace in expression_namespaces.items():
        pl.api.register_expr_namespace(name)(namespace)
    for name, namespace in series_namespaces.items():
        pl.api.register_series_namespace(name)(namespace)
    if entrypoints:
        ep_namespaces = load_setuptools_entrypoints()
        for namespace_type, namespaces in ep_namespaces.items():
            reg_func = namespace_type.to_reg_function()
            for name, namespace in namespaces.items():
                reg_func(name)(namespace)
