from pathlib import Path
from mypy.stubgen import Options as StubOptions, generate_stubs
import sys

from polugins.main import get_entrypoints 
from polugins._types import ExtensionClass
import ast



def generate_polars_stub(output_dir = "typings"):
    modules = [".".join(enum.import_path.parts) for enum in ExtensionClass]
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
            verbose = True, # TODO: change this, but nice for debugging now
            quiet = False,
        )
    generate_stubs(options)

def main():
    generate_polars_stub()

    output_dir = Path("typings")

    entry_points = get_entrypoints()

    for namespace_type, entrypoints in entry_points.items():
        if entrypoints:
            stub_path = (output_dir / namespace_type.import_path).with_suffix(".pyi")
            stub_ast = ast.parse(stub_path.read_text(), type_comments=True)
            new_class_nodes = []
            modules_to_import = set()
            for node in stub_ast.body:
                if isinstance(node, ast.ClassDef) and node.name.casefold() == namespace_type.value.casefold():
                    for ep in entrypoints:
                        annotation_name = ep.value.replace(":", ".")
                        new_node = ast.AnnAssign(
                            target = ast.Name(id=ep.name),
                            annotation = ast.Name(id=annotation_name),
                        simple=1) # no idea what this is
                        new_class_nodes.append(new_node)
                        modules_to_import.add(annotation_name.split(".")[0])
                    node.body.extend(new_class_nodes)
            for name in modules_to_import:
                stub_ast.body.insert(0, ast.Import(names=[ast.Name(id=name)]))
            stub_path.write_text(ast.unparse(stub_ast))

if __name__ == "__main__":
    main()

