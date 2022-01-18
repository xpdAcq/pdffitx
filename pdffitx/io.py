"""The input / output functions related to file system."""
import fabio
from functools import wraps
import numpy as np
from diffpy.srfit.fitbase import Profile
from diffpy.structure import loadStructure
from diffpy.utils.parsers.loaddata import loadData
from pyobjcryst import loadCrystal

from pdffitx import modeling as md
from pdffitx.modeling.fitobjs import MyParser


@wraps(loadCrystal)
def load_crystal(*args, **kwargs):
    return loadCrystal(*args, **kwargs)


@wraps(loadStructure)
def load_structure(*args, **kwargs):
    return loadStructure(*args, **kwargs)


def load_parser(filename: str, meta: dict) -> MyParser:
    """Load data and metadata from the filename into a parser.

    Parameters
    ----------
    filename :
        The name of the data file.

    meta :
        The dictionary of the meta data, like {'qdamp': 0.04, 'qbroad': 0.02}.

    Returns
    -------
    parser :
        The parser that contains the data and metadata.
    """
    parser = MyParser()
    parser.parseFile(filename, meta=meta)
    return parser


def load_profile(filename: str, metadata: dict = None) -> Profile:
    profile = Profile()
    parser = md.MyParser()
    parser.parseFile(filename, metadata)
    profile.loadParsedData(parser)
    return profile


def load_img(img_file: str) -> np.ndarray:
    """Load the img data from the img_file."""
    img = fabio.open(img_file).data
    return img


def load_array(data_file: str, minrows=10, **kwargs) -> np.ndarray:
    """Load data columns from the .txt file and turn columns to rows and return the numpy array."""
    return loadData(data_file, minrows=minrows, **kwargs).T
