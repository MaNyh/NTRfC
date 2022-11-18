=====
NTRfC
=====


Numerical Test Rig for Cascades.


* Free software: MIT license
* Documentation: https://ntrfc.readthedocs.io.
* NTRfC is the base of the (NTRFlows)[https://gitlab.uni-hannover.de/tfd_public/tools/NTRFlows] repository, a workflow for cfd parameter studies


Features
--------

- Easy geometry and postprocessing visualization and manipulation with (pyvista)[https://github.com/pyvista]
- Tested methods and functions for math, timeseries and meshquality-analysis

Dependencies
--------

NTRfC v0.1.0 and any following version is based on Python 3.10.
Only older versions can be used with older versions of Python.

Library-requirements will be installed with the package itself.

Installation
--------

NTRfC is utilizing powerful and complex dependencies like pyvista and gmsh.
It is strongly recommanded to use virtual environments or conda-environments

For more information, see:

- virtualenv: https://pypi.org/project/virtualenv/
- miniconda: https://docs.conda.io/en/latest/miniconda.html
- anaconda: https://docs.anaconda.com/anaconda/install/index.html
- mamba: https://mamba.readthedocs.io/en/latest/installation.html

**Installation from gitlab with pip**

You can install the package with pip from gitlab directly.
Type:

```
pip install git+https://gitlab.uni-hannover.de/tfd_public/tools/NTRfC.git
```


**Installation from source**

After cloning the repository, go to the project root dir and type

```
python setup.py install
```


**Editable installation from source with pip**

After cloning the repository, go to the project root dir and type

```
pip install -e .
```

This way you have NTRfC installed but the code is not installed, but linked to the source-code.
You don't have to reinstall the package after your edits.
This speeds up testing and will lead to less debugging time.

**Singularity releases**

use a singularity container from ntrfc singularity releases:  https://cloud.sylabs.io/library/nyhuma/ntrflows/ntr.sif].
The containers will come with a virtual graphics card and a xvfb display-server, enabling you to render on hpc-systems and any other unprepared system with limited graphics capability.

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
