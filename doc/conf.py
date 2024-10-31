# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# pylint: skip-file

import os
import sys
import logging
import inspect
from sphinx.util.logging import SphinxLoggerAdapter
sys.path.insert(0, os.path.abspath('.'))

# -- Project information -----------------------------------------------------

project = 'SmartSim'
copyright = '2021-2024, Hewlett Packard Enterprise'
author = 'Cray Labs'

try:
    import smartsim
    version = smartsim.__version__
except ImportError:
    version = "0.8.0"

# The full version, including alpha/beta/rc tags
release = version


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx_autodoc_typehints',
    'sphinx.ext.autosectionlabel',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.imgmath',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinxfortran.fortran_domain',
    'sphinxfortran.fortran_autodoc',
    'breathe',
    'nbsphinx',
    'sphinx_copybutton',
    'sphinx_tabs.tabs',
    'sphinx_design',
    'sphinx.ext.mathjax',
    'myst_parser'
]
# sphinx_autodoc_typehints configurations
always_use_bars_union = True
typehints_document_rtype = True
typehints_use_signature = True
typehints_use_signature_return = True
typehints_defaults = 'comma'

autodoc_mock_imports = ["smartredis.smartredisPy"]
suppress_warnings = ['autosectionlabel']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

linkcheck_ignore = [
    'Redis::set_model_multigpu',
]

# The path to the MathJax.js file that Sphinx will use to render math expressions
mathjax_path = 'https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', "**.ipynb_checkpoints", "tutorials/ml_training/surrogate/README.md", "tutorials/online_analysis/lattice/README.md"]

breathe_projects = {
        "c_client":"../smartredis/doc/c_client/xml",
        "fortran_client":"../smartredis/doc/fortran_client/xml",
        "cpp_client":"../smartredis/doc/cpp_client/xml"
        }

fortran_src = [
    "../smartredis/src/fortran/client.F90",
    "../smartredis/src/fortran/dataset.F90"
]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "sphinx_book_theme"

# Check if the environment variable is set to 'True'
if os.environ.get('READTHEDOCS') == "True":
    # If it is, generate the robots.txt file
    with open('./robots.txt', 'w') as f:
        f.write("# Disallow crawling of the Read the Docs URL\nUser-agent: *\nDisallow: /en/")
    html_extra_path = ['./robots.txt']

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

slack_invite ="https://join.slack.com/t/craylabs/shared_invite/zt-nw3ag5z5-5PS4tIXBfufu1bIvvr71UA"
extra_footer = ('Questions? You can contact <a href="mailto:craylabs@hpe.com">contact us</a> or '
                f'<a href="{slack_invite}">join us on Slack!</a>'
                )

html_theme_options = {
    "repository_url": "https://github.com/CrayLabs/SmartSim",
    "use_repository_button": True,
    "use_issues_button": True,
    "extra_footer": extra_footer,
}

# Use a custom style sheet to avoid the sphinx-tabs extension from using
# white background with dark themes.  If sphinx-tabs updates its
# static/tabs.css, this may need to be updated.
html_css_files = ['custom_tab_style.css']
autoclass_content = 'both'
add_module_names = False

nbsphinx_execute = 'never'

from inspect import getsourcefile

# Get path to directory containing this file, conf.py.
DOCS_DIRECTORY = os.path.dirname(os.path.abspath(getsourcefile(lambda: 0)))

def ensure_pandoc_installed(_):
    import pypandoc

    # Download pandoc if necessary. If pandoc is already installed and on
    # the PATH, the installed version will be used. Otherwise, we will
    # download a copy of pandoc into docs/bin/ and add that to our PATH.
    pandoc_dir = os.path.join(DOCS_DIRECTORY, "bin")
    # Add dir containing pandoc binary to the PATH environment variable
    if pandoc_dir not in os.environ["PATH"].split(os.pathsep):
        os.environ["PATH"] += os.pathsep + pandoc_dir
    pypandoc.ensure_pandoc_installed(
        targetfolder=pandoc_dir,
        delete_installer=True,
    )


def setup(app):
    app.connect("builder-inited", ensure_pandoc_installed)

    # Below code from https://github.com/sphinx-doc/sphinx/issues/10219
    def _is_sphinx_logger_adapter(obj):
        return isinstance(obj, SphinxLoggerAdapter)
    class ForwardReferenceFilter(logging.Filter):
        def filter(self, record):
            # Suppress the warning related to forward references
            return "Cannot resolve forward reference in type annotations" not in record.getMessage()

    members = inspect.getmembers(app.extensions['sphinx_autodoc_typehints'].module, _is_sphinx_logger_adapter)
    for _, adapter in members:
        adapter.logger.addFilter(ForwardReferenceFilter())
