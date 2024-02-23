# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys

sys.path.insert(0, os.path.abspath('..'))

project = 'Komanawa.gw_detect_power'
copyright = '2024, Matt Dumont'
author = 'Matt Dumont'
release = 'V2.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.todo', 'sphinx.ext.viewcode', 'sphinx.ext.autodoc',
              'sphinx.ext.autosummary']
extensions.append('autoapi.extension')

# Auto API settings
autoapi_keep_files = True  # Keep the generated files (for debugging)
autoapi_ignore = ['*/lookup_table_inits.py', '*/create_lookup_tables.py']  # Ignore these files
autoapi_python_class_content = 'both'  # Include both the class docstring and the __init__ docstring
autoapi_dirs = ['../src']
autoapi_options = ['members', 'inherited-members', 'show-inheritance', 'show-module-summary', 'imported-members',
                   'show-inheritance-diagram']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- options for autodoc -----------------------------------------------------
add_module_names = False  # Don't add the module name to the class/function name
toc_object_entries_show_parents = 'hide'  # Hide the parent class in the TOC

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
html_sidebars = {'**': [
    # 'globaltoc.html',
    'localtoc.html',
    'searchbox.html']}
