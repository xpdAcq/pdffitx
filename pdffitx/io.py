"""The input / output functions related to file system."""

from diffpy.structure import loadStructure
from pyobjcryst import loadCrystal

from pdffitx.modeling.fitobjs import MyParser

load_crystal = loadCrystal
load_structure = loadStructure


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
