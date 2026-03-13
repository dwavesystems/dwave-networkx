import os
import sys
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# -- General configuration ------------------------------------------------

extensions = [
    'sphinx.ext.autosummary',
    'sphinx.ext.autodoc',
    'sphinx.ext.coverage',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
    'sphinx.ext.ifconfig',
]

autosummary_generate = True

source_suffix = ['.rst']

master_doc = 'index'

# General information about the project.
project = 'D-Wave Graphs'
author = 'D-Wave'
copyright = 'D-Wave'

import dwave.graphs
version = dwave.graphs.__version__
release = dwave.graphs.__version__

language = 'en'

add_module_names = False

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'sdk_index.rst']

linkcheck_retries = 2
linkcheck_anchors = False
linkcheck_ignore = [r'https://scipy.org',   # Looks like it rejects robots
                    r'https://cloud.dwavesys.com/leap',  # redirects, many checks
                    ]

pygments_style = 'sphinx'

todo_include_todos = True

modindex_common_prefix = ['dwave.graphs.']

doctest_global_setup = \
"""
import dwave.graphs
import networkx as nx
"""

# -- Options for HTML output ----------------------------------------------

html_theme = 'pydata_sphinx_theme'
html_theme_options = {
    "collapse_navigation": True,
    "show_prev_next": False,
}
html_sidebars = {"**": ["search-field", "sidebar-nav-bs"]}  # remove ads

intersphinx_mapping = {'python': ('https://docs.python.org/3', None),
    'networkx': ('https://networkx.github.io/documentation/stable/', None),
    'dwave': ('https://docs.dwavequantum.com/en/latest/', None),
    }

# global substitutions
rst_epilog = """
.. |dwave_short| unicode:: D-Wave
"""