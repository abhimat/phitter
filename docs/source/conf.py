# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Phitter'
copyright = '2024, Abhimat K. Gautam'
author = 'Abhimat K. Gautam'
release = '1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    # 'sphinx.ext.autodoc',
    # 'sphinx.ext.autosummary',
    'numpydoc',
    'sphinx.ext.viewcode',
    # 'autoapi.extension',
    'sphinx_copybutton',
    'sphinx_automodapi.automodapi',
    'myst_nb',
    'sphinx_favicon',
    'autodoc2',
]
source_suffix = {
    '.rst': 'restructuredtext',
    '.ipynb': 'myst-nb',
    '.myst': 'myst-nb',
    '.md': 'myst-nb',
}

numpydoc_show_class_members = False
# autoapi_dirs = ['../../phitter']
autodoc2_packages = ['../../phitter']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

nb_execution_mode = "off"


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'alabaster'
html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']
html_css_files = [
    'css/custom.css',
]

html_theme_options = {
    "logo": {
          "image_light": "_static/Phitter.svg",
          "image_dark": "_static/Phitter_dark.svg",
    },
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/abhimat/phitter",
            "icon": "fa-brands fa-square-github",
            "type": "fontawesome",
        },
    ],
    "navbar_align": "left",
}

favicons = [
    "phitter_icon_16.png",
    "phitter_icon_32.png",
    "phitter_icon_100.png",
    "phitter_icon_180.png",
    "phitter_icon.svg",
    {"rel": "apple-touch-icon", "href": "phitter_icon_180.png"},
]