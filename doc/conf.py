# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys
import os
import warnings

sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
project = 'lmlib'
copyright = '2025, lmlib'
author = 'Reto Wildhaber, Frédéric Waldmann'
release = '3.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.coverage",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    'sphinx.ext.napoleon',
    "sphinx.ext.githubpages",
    'sphinx.ext.graphviz',
    'sphinx.ext.inheritance_diagram',
    "matplotlib.sphinxext.plot_directive",
    'sphinx_gallery.gen_gallery',
    "sphinx_design",
]

templates_path = ['templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Extension Options -------------------------------------------------------
# Generate API documentation when building
autosummary_generate = True
add_module_names = False
numpydoc_show_class_members = False
inheritance_graph_attrs = dict(rankdir="TB")

# -- Options for Prolog Keys  ------------------------------------------------

# don't delete the empty line after |br| raw :: html / before <br />
rst_prolog = """
.. |br| raw:: html

   <br />
.. |def_K| replace:: `K` : number of samples |br|
.. |def_k_index| replace:: `k` : sample index |br|
.. |def_Q| replace:: `Q` : output order / number of signal channels |br|
.. |def_S| replace:: `S` : number of signal sets |br|
.. |def_P| replace:: `P` : number of segments |br|
.. |def_M| replace:: `M` : number of ALSSMs |br|
.. |def_J| replace:: `J` : number of ALSSM evaluation indices |br|
.. |def_j_index| replace:: `j` : ALSSM evaluation index |br|
.. |def_N| replace:: `N` : ALSSM system order, corresponding to the number of state variables |br|
.. |def_XS| replace:: `XS` : number of state vectors in a list |br|
.. |def_KS| replace:: `KS` : number of (time) indices in the list |br|
.. |def_JR| replace:: `JR` : index range length |br|
.. |def_L| replace:: `L` : number of dimension of an ND-Alssm/ND-Cost |br|
"""

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_static_path = ['static']

html_sidebars = {
  "path/to/page": [],
}
html_theme_options = {
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["navbar-icon-links"],
    "navbar_persistent": ["search-button"],
    "logo": {
            "text": project + ' ' + release,
            # "image_light": "_static/logo-light.png",
            # "image_dark": "_static/logo-dark.png",
    },
    "icon_links": [
        {"name": "GitHub",
         "url": "https://github.com/lmlib/lmlib",
         "icon": "fa-brands fa-square-github",
         "type": "fontawesome"
         },
    ],
    # "search_bar_text": "Search...",
    # "show_prev_next": False,
    "secondary_sidebar_items": {
            "path/to/page": [],
    },
}


# -- Sphinx Gallery Configuration -----------------------------------------------
from sphinx_gallery.sorting import ExplicitOrder, FileNameSortKey

# html_css_files.append('css/lmlib-gallery.css')
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
    'remove_config_comments': True,
    'write_computation_times': False

}

suppress_warnings = ["config.cache"]
warnings.filterwarnings("ignore", category=UserWarning,
                        message='Matplotlib is currently using agg, which is a'
                                ' non-GUI backend, so cannot show the figure.'
                                '|(\n|.)*is non-interactive, and thus cannot be shown')