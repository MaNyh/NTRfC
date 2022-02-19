=====
NTRfC
=====


.. image:: https://img.shields.io/pypi/v/ntrfc.svg
        :target: https://pypi.python.org/pypi/ntrfc

.. image:: https://img.shields.io/travis/MaNyh/ntrfc.svg
        :target: https://travis-ci.com/MaNyh/ntrfc

.. image:: https://readthedocs.org/projects/ntrfc/badge/?version=latest
        :target: https://ntrfc.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status



Numerical Test Rig for Cascades. A workflow-library for cfd-analysis of cascade-flows


* Free software: MIT license
* Documentation: https://ntrfc.readthedocs.io.

Installation
--------

install using 'pip install ntrfc'
or install using 'python -m setup install'

as a dev use "pip install -e ntrfc" for a linked install.
this way you dont have to reinstall the package after altering the sourcecode.

use 'pip install -v requirements_dev' to install development-packages

Features
--------

database
    - case_templates
        templates can be installed via copying a structure of ascii-files in a directory and defining a schema
postprocessing
    - tbd
preprocessing
    - case_creation
        tools for the definition of cases via templates
    - geometry_creation
        tools for the domain-definition of simulations. under construction
    - mesh_creation
        a library for meshing-scripts
utils
    - dictionaries
        tools for dict-handling specialized for nested dicts
    - filehandling
        handling of different file-formats
    - geometry
        geometry-tools
    - math
        math-library
    - pyvista_utils
        mesh-handling and visualization

    workflows:
        -case_creation
            create a (set of) simulation(s) using a template and configuration files


    examples:
        - gwk_compressor_casecreation

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
