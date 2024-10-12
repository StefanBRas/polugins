import argparse
import ast
import importlib.resources as importlib_resources
from pathlib import Path

from polugins.main import _get_namespaces  # TODO: makes this non private
from typing_extensions import Optional

from polugins_type_gen._ast import (
    parse_from_current_env as _parse_from_current_env,
)
from polugins_type_gen._ast import (
    parse_from_venv as _parse_from_venv,
)


class NoNamespaceRegisteredException(Exception):
    pass


def has_version(version: str) -> bool:
    files = importlib_resources.files("polugins_type_gen")
    return (files / "_stubs" / version).is_dir()


def create_stubs(venv_path: Optional[Path] = None, output_dir: Optional[Path] = None):
    output_dir = output_dir or Path("typings")
    all_namespaces = _get_namespaces()

    if all(namespace == {} for namespace in all_namespaces.values()):
        msg = "No namespaces found. No types will be generated as this is usually an error."
        raise NoNamespaceRegisteredException(msg)

    for extension_class, namespaces in all_namespaces.items():
        if namespaces:
            print(f"Found namespaces for {extension_class.value}:")
            for name, namespace in namespaces.items():
                print(f"  {name}: {namespace}")
            if venv_path:
                stub_ast = _parse_from_venv(venv_path, extension_class)
            else:
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
            print(f"Generated stubs for {extension_class.value} in {output_path.with_suffix('.pyi')}")


def get_argparser():
    parser = argparse.ArgumentParser(
        prog="Polugins Type Gen",
        description="Generates type stubs for polars with added namespaces.",
    )
    parser.add_argument(
        "command",
        help="The command to run",
        choices=["stubs", "version"],
    )
    parser.add_argument(
        "-i",
        "--input",
        help="The input venv to find Polars in. If not selected, uses currently activated venv.",
        required=False,
    )
    parser.add_argument(
        "-o",
        "--output",
        help="The output directory for the generated stubs. Default is ./typings/",
        default="typings",
        required=False,
    )
    return parser


def cli():
    parser = get_argparser()
    args = parser.parse_args()
    if args.command == "stubs":
        create_stubs(
            venv_path=Path(args.input) if args.input else None,
            output_dir=Path(args.output) if args.output else None,
        )
    elif args.command == "version":
        from polugins_type_gen._version import __version__

        print(f"Polugins Type Gen version: {__version__}")
    else:
        msg = "Unknown command. Use `polugins stubs` to generate type stubs."
        raise ValueError(msg)


if __name__ == "__main__":
    cli()
