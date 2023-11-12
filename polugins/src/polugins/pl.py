from polars import *  # noqa: F403

from polugins.main import register_namespaces as _register_namespaces

_register_namespaces(load_entrypoints=True, load_config=True, load_env=True)
