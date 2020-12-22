"""Objects used in the fitting."""
from typing import Union, List, Callable, Tuple

import numpy as np
from diffpy.pdfgetx import PDFGetter, PDFConfig
from diffpy.srfit.fitbase import ProfileGenerator
from diffpy.srfit.pdf import PDFParser
from diffpy.structure import Structure
from numpy import ndarray
from pyobjcryst.crystal import Crystal
from pyobjcryst.molecule import Molecule

__all__ = ["GenConfig", "FunConfig", "ConConfig", "MyParser"]

Stru = Union[Crystal, Molecule, Structure]


def map_stype(mode: str):
    """Map the scattering type in PDFConfig to the stype in the meta in the parser."""
    if mode in ('xray', 'sas'):
        stype = 'X'
    elif mode == 'neutron':
        stype = 'N'
    else:
        raise ValueError(
            "Unknown: scattering type: {}. Allowed values: 'xray', 'neutron', 'sas'.".format(mode)
        )
    return stype


class MyParser(PDFParser):
    """The parser to parse data and meta data."""

    def parseFile(self, filename: str, meta: dict = None):
        """Parse a file and set the _x, _y, _dx, _dy and _meta variables.

        This wipes out the currently loaded data and selected bank number.

        Parameters
        ----------
        filename :
            The name of the file to parse. The meta data will be also read.

        meta :
            The additional meta data like "qdamp" and "qbroad".
        """
        super().parseFile(filename)
        if meta:
            self._meta.update(meta)
        return

    def parseDict(self, data: ndarray, meta: dict = None):
        """Parse the data and meta data from a ndarray and a dictionary.

        Parameters
        ----------
        data : ndarray
            The data. Each row is a data array.

        meta : dict
            The meta data. Valid keys are

                stype       --  The scattering type ("X", "N")
                qmin        --  Minimum scattering vector (float)
                qmax        --  Maximum scattering vector (float)
                qdamp       --  Resolution damping factor (float)
                qbroad      --  Resolution broadening factor (float)
                spdiameter  --  Nanoparticle diameter (float)
                scale       --  Data scale (float)
                temperature --  Temperature (float)
                doping      --  Doping (float)
        """
        if meta is None:
            meta = {}
        if data.shape[0] < 2:
            raise ValueError("Data dimension less than 2.")
        if data.shape[0] > 4:
            raise ValueError("Data dimension larger than 4.")
        self._banks.append(
            [_ for _ in data] + [None] * (4 - data.shape[0])
        )
        self._meta = meta
        return

    def parsePDFGetter(self, pdfgetter: PDFGetter, meta: dict = None):
        """Parse the data and metadta from pdfgetter."""
        if meta is None:
            meta = {}
        data = np.stack(pdfgetter.gr)
        pdfconfig: PDFConfig = pdfgetter.config
        other_meta = {
            'stype': map_stype(pdfconfig.mode),
            'qmin': pdfconfig.qmin,
            'qmax': pdfconfig.qmax,
        }
        meta.update(other_meta)
        self.parseDict(data, meta=meta)
        return


class GenConfig:
    """A configuration class to provide information in the building of PDFGenerator or DebyePDFGenerator. It is
    used by 'make_generator' in 'myscripts.fittingfunction'.

    Attributes
    ----------
    name : str
        The name of the generator.

    structure : Stru
        The structure object. Options are Crystal, Molecule, Structure.

    periodic : bool
        If the structure if periodic. Default if cif or stru, True else False.

    debye : bool
        Use DebyePDFGenerator or PDFGenerator. Default if periodic, False else True.

    ncpu : int
        number of parallel computing cores for the generator. If None, no parallel. Default None.
    """

    def __init__(self, name: str, structure: Stru, periodic: bool = None, debye: bool = None,
                 ncpu: int = None):
        """Initiate the GenConfig."""
        self.name = name
        self.structure = structure
        self.periodic = self.is_periodic(structure) if periodic is None else periodic
        self.debye = not self.periodic if debye is None else debye
        self.ncpu = ncpu

    @staticmethod
    def is_periodic(structure: Stru) -> bool:
        """If Crystal, Structure, return True. If Molecule, return False"""
        if isinstance(structure, Molecule):
            return False
        return True


class FunConfig:
    """Configuration for the characteristic function.

    Attributes
    ----------
        name
            name of the function, also the name in Fitcontribution.
        func
            characteristic function from diffpy cmi.
        argnames
            argument names in the function. it will rename all arguments to avoid conflicts. If None, no renaming.
            If not None, it always starts with "r" when using diffpy characteristic functions. Default None.
    """

    def __init__(self, name: str, func: Callable, argnames: List[str] = None):
        """Initiate Function object."""
        self.name = name
        self.func = func
        self.argnames = argnames


class ConConfig:
    """Configuration for the FitContribution.

    Attributes
    ----------
    name : str
        The name of Fitcontribution.

    data_id : int
        The id of the data. It will be used as a foreign key when the results are saved.

    parser : MyParser
        The parser with parsed data.

    fit_range
        A tuple of (rmin, rmax, rstep) in angstrom for fitting.

    eq
        An equation string for the Fitcontribution. If None, use summation of the partial equation.

    partial_eqs
        The mapping from the phase name to the equation of the PDF. If None, use the eq.

    genconfigs
        A single or a list of GenConfig object. Default empty tuple.

    funconfigs
        A single or a list of FunConfig object. Default empty tuple.

    baselines
        A single or a list of Generator instance of base line. Default empty tuple.

    res_eq
        A string residual equation. Default "chiv".

    weight
        The weight of the contribution. Default 1.
    """

    def __init__(
        self,
        name: str,
        parser: MyParser,
        eq: str,
        fit_range: Tuple[float, float, float],
        genconfigs: List[GenConfig] = None,
        funconfigs: List[FunConfig] = None,
        baselines: List[Callable] = None,
        res_eq: str = "resv",
        weight: float = 1.
    ):
        """Initiate the instance."""
        self.name: str = name
        self.parser: MyParser = parser
        self.fit_range: Tuple[float, float, float] = fit_range
        self.eq: str = eq
        self.genconfigs: List[GenConfig] = genconfigs if genconfigs else list()
        self.funconfigs: List[FunConfig] = funconfigs if funconfigs else list()
        self.baselines: List[ProfileGenerator] = baselines if baselines else list()
        self.res_eq: str = res_eq
        self.weight = weight


