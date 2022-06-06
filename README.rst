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


Numerical Test Rig for Cascades.


* Free software: MIT license
* Documentation: https://ntrfc.readthedocs.io.

Installation
--------

The easiest way to use this repository is to install ntrfc using 'pip install ntrfc'
or install using 'python -m setup install'

when installing from conda, you will run into issues with vtk while using a pip installation of ntrfc.
simply run:
    'conda install -c conda-forge vtk'

as a dev use "pip install -e ntrfc" for a linked install. this way you dont have to reinstall the package after altering the sourcecode.
you might also want to install the dev-libraries in the requirements_dev.txt ('pip install -r requirements_dev.txt')


Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
