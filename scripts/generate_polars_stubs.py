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
    subprocess.check_call([sys.executable, "-m", "pip", "install", f'polars=={version}'])

def get_versions():
    res = subprocess.run([sys.executable, "-m", "pip", "index", "versions",'polars'], capture_output=True, text=True)
    version_line_start = "Available versions: "
    for line in res.stdout.splitlines():
        if line.startswith(version_line_start):
            versions = line.split(version_line_start)[1].split(",")
            break
    versions = [version.strip() for version in versions]
    return versions


def main():
    # TODO: only generate missing type stubs
    versions = get_versions()
    for version in versions:
        output_dir = Path("src", "polugins", "_stubs", version)
        output_dir.mkdir(parents=True, exist_ok=True)
        install_polars(version)
        generate_polars_stub(output_dir)
    generate_all_typestubs() 


if __name__ == "__main__":
    main()
