"""Sphinx configuration for the pymaslow documentation."""

import os
import sys

# Make the package importable for autodoc (src layout). Supports both the
# MaslowTuringTest monorepo layout and the standalone pymaslow repo layout.
for _candidate in ("../../codes/pymaslow/src", "../src"):
    _path = os.path.abspath(_candidate)
    if os.path.isdir(_path):
        sys.path.insert(0, _path)
        break

import pymaslow  # noqa: E402

# -- Project information -----------------------------------------------------

project = "pymaslow"
copyright = "2026, HongSheng Qi"
author = "HongSheng Qi"
version = pymaslow.__version__
release = pymaslow.__version__

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# NumPy-style docstrings
napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_include_init_with_doc = True

autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "undoc-members": True,
    "show-inheritance": True,
}
autodoc_typehints = "description"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "navigation_depth": 3,
    "titles_only": False,
}
html_static_path = []
