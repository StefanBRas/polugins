from typing import Callable, Dict, Optional, Type
from polugins._types import ExtensionClass
import polars as pl
import sys


# TODO: Stolen from pluggy, remember license and stuff
if sys.version_info >= (3, 8):
    from importlib import metadata as importlib_metadata
else:
    # TODO: remember that this is only needed for python < 3.8
    import importlib_metadata  # ignore


def get_entrypoints():
    entry_points = {
        ExtensionClass.EXPR: [],
        ExtensionClass.SERIES: [],
        ExtensionClass.LAZYFRAME: [],
        ExtensionClass.DATAFRAME: [],
    }
    for dist in list(importlib_metadata.distributions()):
        for ep in dist.entry_points:
            if ep.group.startswith("polugins"):
                entry_points[ExtensionClass.from_entrypoint_group(ep.group)].append(ep)
    return entry_points

def load_setuptools_entrypoints():
    """
    TODO: Stolen from pluggy, remember license and stuff
    """
    all_entry_points = get_entrypoints()
    namespaces = {
        ExtensionClass.EXPR: {},
        ExtensionClass.SERIES: {},
        ExtensionClass.LAZYFRAME: {},
        ExtensionClass.DATAFRAME: {},
    }
    for namespace_type, entry_points in all_entry_points.items():
        for ep in entry_points:
            if ep.group.startswith("polugins"):
                namespace = ep.load()
                namespaces[namespace_type][ep.name] = namespace
    return namespaces

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
