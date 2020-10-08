"""The input / output functions related to file system."""
from pathlib import Path
from typing import Dict, Any

import fabio
import pyFAI
import yaml
from diffpy.pdfgetx import PDFConfig, PDFGetter
from diffpy.structure import loadStructure
from diffpy.utils.parsers import loadData
from numpy import ndarray
from pyobjcryst import loadCrystal

from pdfstream.modeling import MyParser

load_crystal = loadCrystal
load_structure = loadStructure


def load_ai_from_poni_file(poni_file: str) -> pyFAI.AzimuthalIntegrator:
    """Initiate the AzimuthalIntegrator using poni file."""
    ai = pyFAI.load(poni_file)
    return ai


def load_ai_from_calib_result(calib_result: dict) -> pyFAI.AzimuthalIntegrator:
    """Initiate the AzimuthalIntegrator using calibration information."""
    ai = pyFAI.AzimuthalIntegrator()
    ai.set_config(calib_result)
    return ai


def load_img(img_file: str) -> ndarray:
    """Load the img data from the img_file."""
    img = fabio.open(img_file).data
    return img


def load_pdfconfig(cfg_file: str) -> PDFConfig:
    """Load the PDFConfig from the processed data file or configuration file."""
    pdfconfig = PDFConfig()
    pdfconfig.readConfig(cfg_file)
    return pdfconfig


def write_out(saving_dir: str, filename: str, pdfgetter: PDFGetter) -> dict:
    """Write out data in pdfgetter into files"""
    data_dirs = {}
    for out_type in pdfgetter.config.outputtypes:
        data_dir = Path(saving_dir).joinpath(out_type)
        if not data_dir.exists():
            data_dir.mkdir()
        data_dirs.update({out_type: data_dir})
    dct = {}
    for out_type in pdfgetter.config.outputtypes:  # out_type in ('iq', 'sq', 'fq', 'gr')
        data_dir = data_dirs.get(out_type)
        out_file = data_dir.joinpath(Path(filename).with_suffix(".{}".format(out_type)).name)
        pdfgetter.writeOutput(str(out_file), out_type)
        dct.update({out_type: str(out_file)})
    return dct


def write_img(filepath: str, img: ndarray, template: str) -> None:
    """Write out the image data as the same type of the template file."""
    temp_img = fabio.open(template)
    temp_img.data = img
    temp_img.save(filepath)
    return


def load_array(data_file: str) -> ndarray:
    """Load data columns from the .txt file and turn columns to rows and return the numpy array."""
    return load_data(data_file).T


def load_dict_from_poni(poni_file: str) -> dict:
    """Turn the poni file to pyFAI readable dictionary."""
    with Path(poni_file).open('r') as f:
        geometry = yaml.safe_load(f)
    return _lower_key(geometry)


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
