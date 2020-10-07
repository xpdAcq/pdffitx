from os import path
from setuptools import setup, find_packages
import sys


# NOTE: This file must remain Python 2 compatible for the foreseeable future,
# to ensure that we error out properly for people with outdated setuptools
# and/or pip.
min_version = (3, 7)
if sys.version_info < min_version:
    error = """
pdffitx does not support Python {0}.{1}.
Python {2}.{3} and above is required. Check your Python version like so:

python3 --version

This may be due to an out-of-date pip. Make sure you have pip >= 9.0.1.
Upgrade pip like so:

pip install --upgrade pip
""".format(*(sys.version_info[:2] + min_version))
    sys.exit(error)

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.rst'), encoding='utf-8') as readme_file:
    readme = readme_file.read()

# parameters for setup
NAME = 'pdffitx'
VERSION = "0.1.0"
DESCRIPTION = "A python package to model atomic pair distribution function (PDF) based on diffpy-cmi."
AUTHOR = "Songsheng Tao"
AUTHOR_EMAIL = 'st3107@columbia.edu'
URL = 'https://github.com/st3107/pdffitx'
LICENSE = "BSD (3-clause)"

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=readme,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    url=URL,
    python_requires='>={}'.format('.'.join(str(n) for n in min_version)),
    packages=find_packages(exclude=['docs', 'tests']),
    entry_points={
        'console_scripts': [
            # 'command = some.module:some_function',
        ],
    },
    include_package_data=True,
    package_data={
        'pdffitx': [
            # When adding files here, remember to update MANIFEST.in as well,
            # or else they will not be included in the distribution on PyPI!
            # 'path/to/data_file',
        ]
    },
    install_requires=[],
    license=LICENSE,
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
    ],
)
