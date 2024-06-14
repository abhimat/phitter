# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Phitter'
copyright = '2024, Abhimat K. Gautam'
author = 'Abhimat K. Gautam'
release = '0.2'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'numpydoc',
    'sphinx.ext.viewcode',
    'autoapi.extension',
    'myst_parser',
    'sphinx_copybutton',
]

autoapi_dirs = ['../../phitter']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



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
