import importlib
from pathlib import Path
from typing import Dict, Type, Union
from polugins._types import ExtensionClass
import sys
import os

try:
    # TODO: Maybe make toml loading an extra
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

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


def load_config_namespaces():
    all_namespaces = {
        ExtensionClass.EXPR: {},
        ExtensionClass.SERIES: {},
        ExtensionClass.LAZYFRAME: {},
        ExtensionClass.DATAFRAME: {},
    }

    def _load_from_toml(toml: dict):
        for namespace_type, namespaces in toml.items():
            extension_class = ExtensionClass(namespace_type)
            for name, namespace_path in namespaces.items():
                module_path, namespace_object_name = namespace_path.split(":")
                namespace_module = importlib.import_module(module_path)
                namespace = getattr(namespace_module, namespace_object_name)
                all_namespaces[extension_class][name] = namespace

    if (polugins_toml := Path("polugins.toml")).exists():
        toml = tomllib.loads(polugins_toml.read_text())
        _load_from_toml(toml)
    if (pyproject := Path("pyproject.toml")).exists():
        pyproject_toml = tomllib.loads(pyproject.read_text())
        toml = pyproject_toml.get("tool", {}).get("polugins")
        if toml:
            _load_from_toml(toml)
    return all_namespaces

def load_env_namespaces():
    all_namespaces = {
        ExtensionClass.EXPR: {},
        ExtensionClass.SERIES: {},
        ExtensionClass.LAZYFRAME: {},
        ExtensionClass.DATAFRAME: {},
    }
    for env_var_name,env_var_value in os.environ.items():
        if env_var_name.casefold().startswith("polugins".casefold()):
            print("here now")
            _, extension_class, name = env_var_name.split("_")
            extension_class = ExtensionClass(extension_class)
            module_path, namespace_object_name = env_var_value.split(":")
            namespace_module = importlib.import_module(module_path)
            namespace = getattr(namespace_module, namespace_object_name)
            all_namespaces[extension_class][name] = namespace

    return all_namespaces


NamespaceDict = Dict[str, Union[Type, str]]


def register_namespaces(
    *,
    lazyframe_namespaces: NamespaceDict = {},
    dataframe_namespaces: NamespaceDict = {},
    expression_namespaces: NamespaceDict = {},
    series_namespaces: NamespaceDict = {},
    load_entrypoints: bool = True,
    load_config: bool = True,
    load_env: bool = True,
):
    for name, namespace in lazyframe_namespaces.items():
        ExtensionClass.LAZYFRAME.register(name, namespace)
    for name, namespace in dataframe_namespaces.items():
        ExtensionClass.DATAFRAME.register(name, namespace)
    for name, namespace in expression_namespaces.items():
        ExtensionClass.EXPR.register(name, namespace)
    for name, namespace in series_namespaces.items():
        ExtensionClass.SERIES.register(name, namespace)
    if load_entrypoints:
        ep_namespaces = load_setuptools_entrypoints()
        for extension_class, namespaces in ep_namespaces.items():
            for name, namespace in namespaces.items():
                extension_class.register(name, namespace)
    if load_config:
        config_namespaces = load_config_namespaces()
        for extension_class, namespaces in config_namespaces.items():
            for name, namespace in namespaces.items():
                extension_class.register(name, namespace)
    if load_env:
        config_namespaces = load_env_namespaces()
        for extension_class, namespaces in config_namespaces.items():
            for name, namespace in namespaces.items():
                extension_class.register(name, namespace)
