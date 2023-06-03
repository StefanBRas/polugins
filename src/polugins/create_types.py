from mypy.stubgen import Options as StubOptions, generate_stubs
import sys
import os
from polugins.main import get_entrypoints, load_setuptools_entrypoints
from polugins._types import Namespace



def generate_polars_stub(modules, output_dir = "typings"):
    options = StubOptions(
            # 'module_file' contains the base class 
            #  onto which dynamic methods are registered
            files = [],
            output_dir=output_dir,
            include_private = True,
            export_less = False,
            # standard params (not auto-defaulted)
            pyversion = sys.version_info[:2],
            interpreter = sys.executable,
            ignore_errors = False,
            parse_only = False,
            no_import = False,
            search_path = ["."],
            doc_dir = '',
            packages = [],
            modules = modules, 
            verbose = True,
            quiet = False,
        )
    generate_stubs(options)

generate_polars_stub(["polars.expr.expr", "polars.lazyframe.frame"])

entry_points = get_entrypoints()
for namespace_type, entrypoints in entry_points.items():
    for ep in entrypoints:
        generate_polars_stub([ep.value.split(":")[0]], "tmp_stubs")

