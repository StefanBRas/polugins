# Polugins

Early PoC for a "plugin" system for polars.

Not production ready - barely even alpha ready. Only uploading it now for discussion and to hog the genius package name.

It's meant to solve two issues with using polars API extensions:

- It's difficult to ensure that the extensions have been registered in all places in a code where you polars is used.

- Extensions breaks static typing.

The idea is to describe a single way to expose and collect API extensions - especially for third party packages - 
and then used this discoverbility to also generate type stubs with the added typing from the extensions.

Users can either call `register_namespaces` themselves or import polars through `polugins.polars` instead.
Lint rules can then be used to enforce that nothing is imported from polars outside of these locations.

This is still a bit annoying no matter what, unless polars does the import natively.

## Implementation

Just a thin wrapper around `polars.api.register_x_namespace` and then using `importlib.metadata` to collect
namespaces from external packages.

## Notes

It's still not entirely clear how an application should register its own namespaces.
Entry points can be used but

- (1) an application might just want to use its own namespaces and not expose them and 
- (2) its a bit annoying, because changes to entrypoints are only registered when the package is installed, even in editable mode (I think).



