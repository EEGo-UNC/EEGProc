# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import tomllib
from pathlib import Path

project = 'EEGProc'
copyright = '2025, Vitor Inserra'
author = 'Vitor Inserra'
# pyproject.toml is the single source of the version.
with open(Path(__file__).resolve().parents[2] / "pyproject.toml", "rb") as _pyproject:
    release = tomllib.load(_pyproject)["project"]["version"]
version = release

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
]

html_theme = "sphinx_rtd_theme"

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = False

typehints_defaults = "comma"
always_document_param_types = True

# Intersphinx: link to external docs
# intersphinx_mapping = {
#     "python": ("https://docs.python.org/3", None),
#     "numpy": ("https://numpy.org/doc/stable/", None),
#     "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
#     "matplotlib": ("https://matplotlib.org/stable/", None),
#     "scipy": ("https://docs.scipy.org/doc/scipy/", None),
#     "pywt": ("https://pywavelets.readthedocs.io/en/latest/", None),
# }

myst_enable_extensions = ["colon_fence", "deflist"]

# Package ``__init__`` modules re-export their submodules' public names.
# Document each object once, where it is defined, so cross-references such as
# ``EEGFrame`` resolve to a single target.
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
    "ignore-module-all": True,
}
