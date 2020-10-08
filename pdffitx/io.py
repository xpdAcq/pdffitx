"""The input / output functions related to file system."""
from typing import Dict, Any

import yaml
from diffpy.structure import loadStructure
from diffpy.utils.parsers import loadData
from pyobjcryst import loadCrystal

from pdffitx.modeling import MyParser

load_crystal = loadCrystal
load_structure = loadStructure


def _lower_key(dct: Dict[str, Any]) -> Dict[str, Any]:
    """Return dictionary with all keys in lower case."""
    return {key.lower(): value for key, value in dct.items()}


load_data = loadData


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
