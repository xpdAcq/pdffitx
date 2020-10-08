============
Installation
============

Prerequisites
-------------

Install `Anaconda <https://docs.conda.io/projects/conda/en/latest/user-guide/install/>`_.

After conda is installed, at the commnad line::

    conda config --append channels diffpy
    conda config --append channels conda-forge

(Optional) Get the .whl file of `PDFgetX <https://www.diffpy.org/products/pdfgetx.html>`_. This package is used
to transform the XRD data to PDF data.

General Installation
--------------------

Users can install the `pdffitx` using conda. It is suggested to create a new environment for it.

At the command line::

    conda create -n pdffitx_env -c st3107 pdfstream

The ``pdfstream_env`` in the command is the name of the environment. It can be changed to any name.

Activate the environment::

    conda activate pdffitx_env

(Optional) Install the `diffpy.pdfgetx` using .whl file::

    python -m pip install <path to .whl file>

Change the ``<path to .whl file>`` to the path of the .whl file on your computer.

Before using the `pdffitx`, remember to activate the environment::

    conda activate pdffitx_env

Development Installation
------------------------

**Fork** and clone the github repo and change the current directory::

    git clone https://github.com/<your account>/pdffitx

Remember to change ``<your account>`` to the name of your github account.

Change directory::

    cd pdffitx

Create an environment with all the requirements::

    conda create -n pdffitx_env --file requirements/build.txt --file requirements/run.txt --file requirements/test.txt

(Optional) For the maintainer, install the packages for building documents and releasing the software::

    conda install -n pdffitx_env --file requirements/docs.txt --file requirements/release.txt

Activate the environment::

    conda activate pdffitx_env

Install the `diffpy.pdfgetx` using .whl file::

    python -m pip install <path to .whl file>

Install the `twine` for pypi release::

    python -m pip install twine

Change the ``<path to .whl file>`` to the path of the .whl file on your computer.

Install the `pdffitx` in development mode::

    python -m pip install -e .

