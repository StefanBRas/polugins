import sys
from pathlib import Path

from mypy.stubgen import Options as StubOptions
from mypy.stubgen import generate_stubs

IMPORT_PATHS = [
    Path("polars", "expr", "expr"),
    Path("polars", "series", "series"),
    Path("polars", "lazyframe", "frame"),
    Path("polars", "dataframe", "frame"),
]


if __name__ == "__main__":
    output_dir = sys.argv[1]
    modules = [".".join(import_path.parts) for import_path in IMPORT_PATHS]
    include_docstrings = sys.argv[2] == "true"
    options = StubOptions(
        inspect=True,
        # 'module_file' contains the base class
        #  onto which dynamic methods are registered
        files=[],
        output_dir=output_dir,
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
        include_docstrings=include_docstrings,
    )
    generate_stubs(options)
