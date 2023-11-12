import os
from pathlib import Path
from typing import Dict, Type, Union

from polugins._types import ExtensionClass

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

from importlib import metadata as importlib_metadata


def _get_entrypoint_namespaces():
    namespaces = {
        ExtensionClass.EXPR: {},
        ExtensionClass.SERIES: {},
        ExtensionClass.LAZYFRAME: {},
        ExtensionClass.DATAFRAME: {},
    }
    for dist in list(importlib_metadata.distributions()):
        for ep in dist.entry_points:
            if ep.group.startswith("polugins"):
                extension_class = ExtensionClass.from_entrypoint_group(ep.group)
                namespaces[extension_class][ep.name] = ep.value
    return namespaces


def _get_config_namespaces():
    all_namespaces = {
        ExtensionClass.EXPR: {},
        ExtensionClass.SERIES: {},
        ExtensionClass.LAZYFRAME: {},
        ExtensionClass.DATAFRAME: {},
    }

    def _get_from_toml(toml: dict):
        for namespace_type, namespaces in toml.items():
            extension_class = ExtensionClass(namespace_type)
            for name, namespace_path in namespaces.items():
                all_namespaces[extension_class][name] = namespace_path

    if (polugins_toml := Path("polugins.toml")).exists():
        toml = tomllib.loads(polugins_toml.read_text())
        _get_from_toml(toml)
    if (pyproject := Path("pyproject.toml")).exists():
        pyproject_toml = tomllib.loads(pyproject.read_text())
        toml = pyproject_toml.get("tool", {}).get("polugins")
        if toml:
            _get_from_toml(toml)
    return all_namespaces


def _get_env_namespaces():
    all_namespaces = {
        ExtensionClass.EXPR: {},
        ExtensionClass.SERIES: {},
        ExtensionClass.LAZYFRAME: {},
        ExtensionClass.DATAFRAME: {},
    }
    for env_var_name, env_var_value in os.environ.items():
        if env_var_name.casefold().startswith("polugins".casefold()):
            _, extension_class, name = env_var_name.split("_")
            extension_class = ExtensionClass(extension_class)
            all_namespaces[extension_class][name] = env_var_value
    return all_namespaces


NamespaceDict = Dict[str, Union[Type, str]]


def _get_namespaces(
    load_entrypoints: bool = True,
    load_config: bool = True,
    load_env: bool = True,
):
    all_namespaces = {
        ExtensionClass.EXPR: {},
        ExtensionClass.SERIES: {},
        ExtensionClass.LAZYFRAME: {},
        ExtensionClass.DATAFRAME: {},
    }
    if load_entrypoints:
        ep_namespaces = _get_entrypoint_namespaces()
        for extension_class, namespaces in ep_namespaces.items():
            for name, namespace in namespaces.items():
                all_namespaces[extension_class][name] = namespace
    if load_config:
        config_namespaces = _get_config_namespaces()
        for extension_class, namespaces in config_namespaces.items():
            for name, namespace in namespaces.items():
                all_namespaces[extension_class][name] = namespace
    if load_env:
        config_namespaces = _get_env_namespaces()
        for extension_class, namespaces in config_namespaces.items():
            for name, namespace in namespaces.items():
                all_namespaces[extension_class][name] = namespace
    return all_namespaces


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
    other_namespaces = _get_namespaces(load_entrypoints, load_config, load_env)
    for extension_class, namespaces in other_namespaces.items():
        for name, namespace in namespaces.items():
            extension_class.register(name, namespace)
