import os
import sys

# Add the project directory to the Python path
sys.path.insert(0, os.path.abspath(".."))

# Project information
project = "ntrfc"
author = "Your Name"

# -- General configuration ------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
]

templates_path = ["_templates"]
source_suffix = ".rst"
master_doc = "index"
language = "en"

# -- Options for HTML output ----------------------------------------------

html_theme = "alabaster"
html_theme_options = {
    "logo": "logo.png",
    "github_user": "your-github-username",
    "github_repo": "your-github-repo",
    "github_banner": True,
    "show_powered_by": False,
}

html_static_path = ["_static"]
html_css_files = ["custom.css"]

# -- Options for extensions ------------------------------------------------

# Add any additional extension-specific configuration here
# For example, to configure autodoc:
# autodoc_default_options = {
#     'members': True,
#     'undoc-members': True,
# }

# -- Options for Sphinx build ---------------------------------------------

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
pygments_style = "sphinx"
html_show_sourcelink = False
html_show_sphinx = False
html_favicon = "_static/favicon.ico"

# -- Options for HTMLHelp output ------------------------------------------

htmlhelp_basename = "ntrfcdoc"

# -- Options for LaTeX output ---------------------------------------------

latex_elements = {}

latex_documents = [
    (master_doc, "ntrfc.tex", "ntrfc Documentation", "Your Name", "manual"),
]

# -- Options for manual page output ---------------------------------------

man_pages = [
    (master_doc, "ntrfc", "ntrfc Documentation", [author], 1)
]

# -- Options for Texinfo output -------------------------------------------

texinfo_documents = [
    (
        master_doc,
        "ntrfc",
        "ntrfc Documentation",
        author,
        "ntrfc",
        "One line description of project.",
        "Miscellaneous",
    ),
]
