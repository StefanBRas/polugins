"""Nox sessions.
From https://github.com/cjolowicz/cookiecutter-hypermodern-python-instance
"""
from pathlib import Path
from typing import Iterable

import nox

python_versions = ["3.11", "3.10", "3.9", "3.8"]


def install(session: nox.Session, *, groups: Iterable[str], root: bool = True) -> None:
    """Install the dependency groups using Poetry.

    This function installs the given dependency groups into the session's
    virtual environment. When ``root`` is true (the default), the function
    also installs the root package and its default dependencies.

    To avoid an editable install, the root package is not installed using
    ``poetry install``. Instead, the function invokes ``pip install .``
    to perform a PEP 517 build.

    Args:
        session: The Session object.
        groups: The dependency groups to install.
        root: Install the root package.
    """
    session.run_always(
        "poetry",
        "install",
        "--no-root",
        "--sync",
        "--{}={}".format("only" if not root else "with", ",".join(groups)),
        external=True,
    )
    breakpoint()
    #if root:
    #    session.install(".")

@nox.session(python=python_versions)
def test(session: nox.Session) -> None:
    """Run the test suite."""
    install(session, groups=['testing'], root=True)

    try:
        session.run("coverage","erase")
        session.run("pytest","--cov=src/polugins",  *session.posargs)
    finally:
        if session.interactive:
            session.notify("coverage", posargs=[])


@nox.session(python=python_versions[0])
def coverage(session: nox.Session) -> None:
    """Run the test suite."""
    ...

