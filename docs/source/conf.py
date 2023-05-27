import os
import sys
sys.path.insert(0, os.path.abspath('../../src'))

project = 'py-flowsom'

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon"
]
napoleon_numpy_docstring = False  # Default is True

import sphinx_rtd_theme

html_theme = "sphinx_rtd_theme"
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
