import importlib
from typing import Callable, Dict, Optional, Type, Union
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

NamespaceDict = Dict[str, Union[Type, str]]

def register_namespaces(
    lazyframe_namespaces: NamespaceDict = {},
    dataframe_namespaces: NamespaceDict = {},
    expression_namespaces: NamespaceDict = {},
    series_namespaces: NamespaceDict = {},
    entrypoints: bool = True,
):
    for name, namespace in lazyframe_namespaces.items():
        ExtensionClass.LAZYFRAME.register(name, namespace)
    for name, namespace in dataframe_namespaces.items():
        ExtensionClass.DATAFRAME.register(name, namespace)
    for name, namespace in expression_namespaces.items():
        ExtensionClass.EXPR.register(name, namespace)
    for name, namespace in series_namespaces.items():
        ExtensionClass.SERIES.register(name, namespace)
    if entrypoints:
        ep_namespaces = load_setuptools_entrypoints()
        for extension_class, namespaces in ep_namespaces.items():
            for name, namespace in namespaces.items():
                extension_class.register(name, namespace)
