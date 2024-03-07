"""Namespaces for testing
Yes, should go in a test packages.
Here for now to test a package importing from itself
"""

import polars as pl


class CustomNamespace:
    def __init__(self, ldf: pl.LazyFrame):
        self._ldf = ldf

    def custom_method(self, x: int) -> pl.LazyFrame:
        """my docstring"""
        return self._ldf


class CustomNamespaceByPath:
    def __init__(self, ldf: pl.LazyFrame):
        self._ldf = ldf

    def custom_method(self, x: int) -> pl.LazyFrame:
        """my docstring"""
        return self._ldf


class EntryPointNameSpace:
    def __init__(self, ldf: pl.LazyFrame):
        self._ldf = ldf

    def custom_method(self, x: int) -> pl.LazyFrame:
        """my docstring"""
        return self._ldf


class PyProjectNameSpace:
    def __init__(self, ldf: pl.LazyFrame):
        self._ldf = ldf

    def custom_method(self, x: int) -> pl.LazyFrame:
        """my docstring"""
        return self._ldf


class ConfigNameSpace:
    def __init__(self, ldf: pl.LazyFrame):
        self._ldf = ldf

    def custom_method(self, x: int) -> pl.LazyFrame:
        """my docstring"""
        return self._ldf


class EnvNameSpace:
    def __init__(self, ldf: pl.LazyFrame):
        self._ldf = ldf

    def custom_method(self, x: int) -> pl.LazyFrame:
        """my docstring"""
        return self._ldf
