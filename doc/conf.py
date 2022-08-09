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

sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------

project = 'lmlib Doc'
copyright = '2022, lmlib'
author = 'Reto Wildhaber, Frédéric Waldmann'

# The full version, including alpha/beta/rc tags
release = '2.0 CFR 1'

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
              # 'nbsphinx',
              ]

autosummary_generate = True  # Turn on sphinx.ext.autosummary
# autoclass_content = 'class'
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


# -- Sphinx Gallery Configuration -----------------------------------------------
from sphinx_gallery.sorting import ExplicitOrder, FileNameSortKey
sphinx_gallery_conf = {
    'examples_dirs': ['../examples', '../coding'],   # path to your example scripts
    'gallery_dirs': ['_gallery_examples', '_gallery_coding'],  # path to where to save gallery generated output
    'subsection_order': ExplicitOrder([
                                        '../examples/12-filtering',
                                        '../examples/40-app-changepoint-detection',
                                        '../examples/11-detection',
                                        # '../examples/13-lssm-costs-others',
                                        '../examples/20-polynomials-basics',
                                        '../examples/50-convolution',
                                        '../examples/21-polynomials-calculus',
                                        '../examples/70-localized-polynomials',
                                        # '../examples/30-utils',
                                        '../coding/10-windowed-state-space-filters-basic',
                                        '../coding/13-backend',
                                        '../coding/20-polynomials-basics'
    ]),
    'within_subsection_order': FileNameSortKey,
    'filename_pattern': '(/example-|/fig-)',
    'ignore_pattern': r'(L|draft_)',
    # directory where function/class granular galleries are stored
    'backreferences_dir'  : '_gallery_api/',
    # Modules for which function/class level galleries are created. In
    # this case sphinx_gallery and numpy in a tuple of strings.
    'doc_module'          : ('lmlib',),
    'reference_url': {
         # The module you locally document uses None
        'lmlib': None,
    },
    'thumbnail_size': (500, 400),
    'default_thumb_file': 'static/gallery/default-thumbnail.png',    
}


# -- Options for interphinx reference third-party libs -----------------------

intersphinx_mapping = {'python': ('https://docs.python.org/3', None),
                       'numpy': ('https://numpy.org/doc/stable/', None),
                       'scipy': ('https://docs.scipy.org/doc/scipy/reference', None),
                       'matplotlib': ('https://matplotlib.org', None),
                       }

# -- Options for Pygments (syntax highlighting) ------------------------------

highlight_language = 'python3' # The name of the Pygments (syntax highlighting) style to use.

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

"""

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'pydata_sphinx_theme'


html_sidebars = {
  "**": []
}
html_theme_options = {
    "navbar_end": ["navbar-icon-links.html", "search-field.html"],
    "icon_links": [
        {"name": "GitHub",
         "url": "https://github.com/lmlib/lmlib",
         "icon": "fab fa-github-square",
         "type": "fontawesome"
         },
        {"name" : "Install",
         "url" : ""}
    ]
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['static']

html_css_files = [
    'css/lmlib.css',
]
