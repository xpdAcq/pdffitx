"""Core of the PDFfitx."""
from typing import Dict, Union

from diffpy.srfit.equation.builder import EquationFactory
from diffpy.srfit.fitbase import FitRecipe, FitContribution
from diffpy.srfit.fitbase import FitResults
from diffpy.srfit.pdf import PDFGenerator, DebyePDFGenerator


class MyContribution(FitContribution):
    """The FitContribution with augmented features."""

    @property
    def generators(self) -> Dict[str, Union[PDFGenerator, DebyePDFGenerator]]:
        return self._generators

    @property
    def eqfactory(self) -> EquationFactory:
        return self._eqfactory

    @property
    def xname(self) -> str:
        return self._xname


class MyRecipe(FitRecipe):
    """The FitRecipe interface with augmented features."""

    @property
    def contributions(self) -> Dict[str, MyContribution]:
        return self._contributions


class MyFitResults(FitResults):
    """The augmented fit result interface."""
    pass
