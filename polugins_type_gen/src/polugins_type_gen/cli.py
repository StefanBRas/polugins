import ast
import importlib.resources as importlib_resources
import sys
from pathlib import Path

from polugins.main import _get_namespaces  # TODO: makes this non private

from polugins_type_gen._ast import parse_from_current_env as _parse_from_current_env


class NoNamespaceRegisteredException(Exception):
    pass


def has_version(version: str) -> bool:
    files = importlib_resources.files("polugins_type_gen")
    return (files / "_stubs" / version).is_dir()


def create_stubs():
    output_dir = Path("typings")
    all_namespaces = _get_namespaces()

    if all(namespace == {} for namespace in all_namespaces.values()):
        msg = "No namespaces found. No types will be generated as this is usually an error."
        raise NoNamespaceRegisteredException(msg)

    for extension_class, namespaces in all_namespaces.items():
        if namespaces:
            stub_ast = _parse_from_current_env(extension_class)
            new_class_nodes = []
            module_imports = set()
            for node in stub_ast.body:
                if isinstance(node, ast.ClassDef) and node.name.casefold() == extension_class.value.casefold():
                    for name, namespace in namespaces.items():
                        module_path, class_name = namespace.split(":")
                        new_node = ast.AnnAssign(
                            target=ast.Name(id=name),
                            annotation=ast.Name(id=f"{module_path}.{class_name}"),
                            simple=1,
                        )  # no idea what `simple` does, but it breaks without it
                        new_class_nodes.append(new_node)
                        module_imports.add(module_path)
                    node.body.extend(new_class_nodes)

            new_module_imports = [ast.Import(names=[ast.alias(name=module_name)]) for module_name in module_imports]
            future_import_index = next(
                i
                for i, node in enumerate(stub_ast.body)
                if isinstance(node, ast.ImportFrom) and node.module == "__future__"
            )

            stub_ast.body[(future_import_index + 1) : (future_import_index + 1)] = new_module_imports
            output_path = output_dir / extension_class.import_path
            output_path.parent.mkdir(exist_ok=True, parents=True)
            output_path.with_suffix(".pyi").write_text(ast.unparse(stub_ast))


def cli():
    if len(sys.argv) == 1:
        from polugins_type_gen._version import __version__

        print(f"Polugins Type Gen version: {__version__}")
    elif sys.argv[1] == "stubs":
        print("generating stubs at ./typings/")
        create_stubs()
    else:
        print("Use `polugins stubs` to generate type stubs.")


if __name__ == "__main__":
    cli()
