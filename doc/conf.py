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
import os
import sys
import shlex
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = 'Chemometric Extensions'
copyright = '2020, Leslie Simon, Group of Prof. Dr. P. Weller, Institute of Instrumental Analysis, Hochschule Mannheim-University of applied sciences'
author = 'Leslie Simon, Group of Prof. Dr. P. Weller, Institute of Instrumental Analysis, Hochschule Mannheim-University of applied sciences'

# The full version, including alpha/beta/rc tags
version = '0.0.1'
release = '0.0.1'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

source_suffix = ['.rst', '.md']
source_parsers = {".md": "recommonmark.parser.CommonMarkParser"}
# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

master_doc = 'index'
# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'nature'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

language = None

exclude_patterns = ['build']
pygments_style = 'sphinx'
todo_include_todos = False
htmlhelp_basename = 'Orange3PCAExtensionAdd-ondoc'

man_pages = [
    (master_doc, 'orange3PCAExtensionadd-on', 'Orange3 PCAExtension Add-on Documentation',
     [author], 1)
]

texinfo_documents = [
  (master_doc, 'Orange3PCAExtensionAdd-on', 'Orange3 PCAExtension Add-on Documentation',
   author, 'Orange3PCAExtensionAdd-on', 'One line description of project.',
   'Miscellaneous'),
]