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
import subprocess
import warnings

sys.path.insert(0, os.path.abspath('..'))

# # -- Run additional python scripts forehand ---------------------------------
# folders_pr = ['static/']
#
# for folder in folders_pr:
#     py_files = [f for f in os.listdir(folder) if (os.isfile(os.join(folder, f)) and f.endswith('.py'))]
#     for py_file in py_files:
#         try:
#             print(f"running: {py_file}")
#             subprocess.call([py_file])
#         except:
#             print("An exception occurred while running the script")




# -- Project information -----------------------------------------------------

project = 'lmlib'
copyright = '2024, lmlib'
author = 'Reto Wildhaber, Frédéric Waldmann'

# The full version, including alpha/beta/rc tags
release = '2.1.2'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.autosummary',
              'sphinx.ext.napoleon',
              'sphinx.ext.intersphinx',
              'sphinx.ext.mathjax',
              'sphinx.ext.doctest',
              'matplotlib.sphinxext.plot_directive',
              'sphinx_gallery.gen_gallery',
              'sphinx_design',
              # 'nbsphinx',
              ]

autosummary_generate = True  # Turn on sphinx.ext.autosummary
# autoclass_content = 'class'
# autoclass_content = 'both' # to show constructors
# add_module_names = False  # hides the module path to the class in the class title
# napoleon_use_rtype = False  # includes the return type into the return variable description
# napoleon_use_admonition_for_examples = False
napoleon_use_ivar = False  # attribute display type
napoleon_include_init_with_doc = False  # if false hides __init__() doc at top
# napoleon_use_param = False
# napoleon_use_keyword = False
# napoleon_preprocess_types = True
numpydoc_show_inherited_class_members = False
# Add any paths that contain templates here, relative to this directory.
templates_path = ['templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for interphinx reference third-party libs -----------------------

intersphinx_mapping = {'python': ('https://docs.python.org/3', None),
                       'numpy': ('https://numpy.org/doc/stable/', None),
                       'scipy': ('https://docs.scipy.org/doc/scipy/reference', None),
                       'matplotlib': ('https://matplotlib.org', None),
                       }

# -- Options for Pygments (syntax highlighting) ------------------------------

highlight_language = 'python3'  # The name of the Pygments (syntax highlighting) style to use.

# -- Global reference list ---------------------------------------------------

# don't delete the empty line after |br| raw :: html / before <br />
rst_prolog = """
.. |br| raw:: html

   <br />
.. |def_K| replace:: `K` : number of samples |br|
.. |def_k_index| replace:: `k` : sample index |br|
.. |def_L| replace:: `L` : output order / number of signal channels |br|
.. |def_S| replace:: `S` : number of signal sets |br|
.. |def_P| replace:: `P` : number of segments |br|
.. |def_M| replace:: `M` : number of ALSSMs |br|
.. |def_J| replace:: `J` : number of ALSSM evaluation indices |br|
.. |def_j_index| replace:: `j` : ALSSM evaluation index |br|
.. |def_N| replace:: `N` : ALSSM system order, corresponding to the number of state variables |br|
.. |def_XS| replace:: `XS` : number of state vectors in a list |br|
.. |def_KS| replace:: `KS` : number of (time) indices in the list |br|
.. |def_JR| replace:: `JR` : index range length |br|
.. |def_Q| replace:: `Q` : Polynomial Order |br|
"""

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'pydata_sphinx_theme'
html_theme_options = {
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["navbar-icon-links"],
    "navbar_persistent": ["search-button"],
    "icon_links": [
        {"name": "GitHub",
         "url": "https://github.com/lmlib/lmlib",
         "icon": "fa-brands fa-square-github",
         "type": "fontawesome"
         },
    ],
    "search_bar_text": "Search...",
    "show_prev_next": False
}
html_context = {
   "default_mode": "light"
}
html_static_path = ['static']
html_css_files = ['css/lmlib.css']




# -- Sphinx Gallery Configuration ---
from sphinx_gallery.sorting import ExplicitOrder, FileNameSortKey

html_css_files.append('css/lmlib-gallery.css')
sphinx_gallery_conf = {
    'examples_dirs': ['../examples', '../coding'],
    'gallery_dirs': ['_gallery_examples', '_gallery_coding'],
    'within_subsection_order': FileNameSortKey,
    'filename_pattern': '(/example-|/fig-)',
    'ignore_pattern': r'L',
    'doc_module': ('lmlib',),
    'reference_url': {
        'lmlib': None,
    },
    'thumbnail_size': (400, 300),
    'default_thumb_file': 'static/gallery/default-thumbnail.png',
    'show_memory': False,

}

warnings.filterwarnings("ignore", category=UserWarning,
                        message='Matplotlib is currently using agg, which is a'
                                ' non-GUI backend, so cannot show the figure.'
                                '|(\n|.)*is non-interactive, and thus cannot be shown')
