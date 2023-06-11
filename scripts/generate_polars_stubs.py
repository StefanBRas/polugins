from pathlib import Path
from mypy.stubgen import Options as StubOptions, generate_stubs
import sys
import os
import hashlib
import subprocess
import sys
from dataclasses import dataclass

from polugins.main import get_entrypoints
from polugins._types import ExtensionClass
import ast


def generate_polars_stub(output_dir: Path):
    modules = [".".join(enum.import_path.parts) for enum in ExtensionClass]
    options = StubOptions(
        # 'module_file' contains the base class
        #  onto which dynamic methods are registered
        files=[],
        output_dir=str(output_dir),
        include_private=True,
        export_less=False,
        # standard params (not auto-defaulted)
        pyversion=sys.version_info[:2],
        interpreter=sys.executable,
        ignore_errors=False,
        parse_only=False,
        no_import=False,
        search_path=["."],
        doc_dir="",
        packages=[],
        modules=modules,
        verbose=True,  # TODO: change this, but nice for debugging now
        quiet=False,
    )
    generate_stubs(options)


def install_polars(version: str):
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", f"polars=={version}"]
    )


def get_versions():
    res = subprocess.run(
        [sys.executable, "-m", "pip", "index", "versions", "polars"],
        capture_output=True,
        text=True,
    )
    version_line_start = "Available versions: "
    for line in res.stdout.splitlines():
        if line.startswith(version_line_start):
            versions = line.split(version_line_start)[1].split(",")
            break
    versions = [version.strip() for version in versions]
    return versions


common_transformation = {
    ExtensionClass.DATAFRAME: {  # imports: paramspec, generic
        "class DataFrame:": "class DataFrame(Generic[P]):",
        "P: Incomplete": "ParamSpec('P')",
        "_df: Incomplete": "_df: PyDataFrame",
        "from _typeshed import Incomplete": "from typing_extensions import ParamSpec",
    }
}


def clean_types(path: Path):
    extensions_class = ExtensionClass(path.parts[5])
    stub_content = path.read_text()
    match extensions_class:
        case ExtensionClass.DATAFRAME:
            if (txt := "P: Incomplete") in stub_content:
                stub_content = (
                    "from typing_extensions import ParamSpec, Generic\n"
                    + stub_content.replace(txt, "P = ParamSpec('P')").replace(
                        "class DataFrame", "class DataFrame(Generic[P])"
                    )
                )
            stub_content = stub_content.replace("_df: Incomplete", "_df: PyDataFrame")
        case ExtensionClass.LAZYFRAME:
            if (txt := "P: Incomplete") in stub_content:
                stub_content = (
                    "from typing_extensions import ParamSpec, Generic\n"
                    + stub_content.replace(txt, "P = ParamSpec('P')").replace(
                        "class LazyFrame", "class LazyFrame(Generic[P])"
                    )
                )
            stub_content = stub_content.replace("_ldf: Incomplete", "_ldf: PyLazyFrame")
        case ExtensionClass.EXPR:
            if (txt := "P: Incomplete") in stub_content:
                stub_content = (
                    "from typing_extensions import ParamSpec, Generic\n"
                    + stub_content.replace(txt, "P = ParamSpec('P')").replace(
                        "class Expr", "class Expr(Generic[P])"
                    )
                )
        case ExtensionClass.SERIES:
            pass
    stub_content = stub_content.replace("from _typeshed import Incomplete", "")

    path.with_suffix("").write_text(stub_content)


def is_incomplete(path: Path):
    assert "Incomplete" in path.read_text()


def main():
    # TODO: only generate missing type stubs
    #versions = get_versions()
    #for version in versions:
    #    output_dir = Path("src", "polugins", "_stubs", version)
    #    output_dir.mkdir(parents=True, exist_ok=True)
    #    install_polars(version)
    #    generate_polars_stub(output_dir)
    for version_dir in Path("src", "polugins", "_stubs").glob("*"):
        for extension_class in ExtensionClass:
            stub_path = version_dir / extension_class.import_path.with_suffix(".pyi")
            clean_types(stub_path)


if __name__ == "__main__":
    main()
