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
sys.path.insert(0, os.path.abspath('.'))

# -- Project information -----------------------------------------------------

project = 'SmartSim'
copyright = '2021-2024, Hewlett Packard Enterprise'
author = 'Cray Labs'

try:
    import smartsim
    version = smartsim.__version__
except ImportError:
    version = "0.6.2"

# The full version, including alpha/beta/rc tags
release = version


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
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
    'sphinx_tabs.tabs'
]

suppress_warnings = ['autosectionlabel']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', "**.ipynb_checkpoints"]

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
