# Polugins

Early PoC for a "plugin" system for polars.

Not production ready - barely even alpha ready. Only uploading it now for discussion and to hog the genius package name.

It's meant to solve two issues with using polars API extensions:

- It's difficult to ensure that extensions everywhere in a code base.

- Extensions breaks static typing. (Not implemented yet)

The idea is to describe a single way to expose and collect API extensions - especially for third party packages - 
and then used this discoverbility to also generate type stubs with the added typing from the extensions.

Users can either call `register_namespaces` themselves or import polars through `polugins.polars` instead.
Lint rules can then be used to enforce that nothing is imported from polars outside of these locations.

This is still a bit annoying no matter what, unless polars does the import natively.

## Package example

See `tests/pkgs/example_package` for how an package could expose namespaces.

It's done using entry points, here specified in the `pyproject.toml`.

## Usage example

`example-package` exposes a `LazyFrame` namespace called `external`. After installing it, users
can either do

```python
from polugins.main import register_namespaces
import polars as pl

register_namespaces(entrypoints=True)

pl.LazyFrame().external.some_method(x=1)
```

or

```python
import polugins.polars as pl

pl.LazyFrame().external.some_method(x=1)
```

Which automagically registers namespaces from entry points.

## Implementation

Just a thin wrapper around `polars.api.register_x_namespace` and then using `importlib.metadata` to collect
namespaces from external packages.

## Notes

It's still not entirely clear how an application should register its own namespaces.
Entry points can be used but

- (1) an application might just want to use its own namespaces and not expose them and 
- (2) its a bit annoying, because changes to entrypoints are only registered when the package is installed, even in editable mode (I think).



