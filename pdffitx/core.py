"""Core of the PDFfitx."""
from typing import Dict, Union

from diffpy.srfit.equation.builder import EquationFactory
from diffpy.srfit.fitbase import FitRecipe, FitContribution
from diffpy.srfit.fitbase import FitResults
from diffpy.srfit.pdf import PDFGenerator, DebyePDFGenerator

from pdffitx.modeling.adding import initialize
from pdffitx.modeling.creating import create
from pdffitx.modeling.main import optimize, view_fits


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

    @classmethod
    def create(
        cls,
        name: str,
        data: MyParser,
        arange: tp.Tuple[float, float, float],
        equation: str,
        functions: tp.Dict[str, tp.Callable],
        structures: tp.Dict[str, S],
        ncpu: int = None
    ):
        return create(
            name=name,
            data=data,
            arange=arange,
            equation=equation,
            functions=functions,
            structures=structures,
            ncpu=ncpu
        )

    def initialize(
        self,
        scale: bool = True,
        delta: tp.Union[str, None] = "2",
        lat: tp.Union[str, None] = "s",
        adp: tp.Union[str, None] = "a",
        xyz: tp.Union[str, None] = "s",
        params: tp.Union[None, str, tp.List[str]] = "a"
    ):
        """Initialize a single-contribution recipe with the variables.

        The variables will be constrained, created and added according to the mode indicated by the arguments.
        If an argument is None, nothing will be done. It is assumed that the recipe only has one contribution
        and all variables in the generators in that contribution will be initialized in the same way.

        The name for each parameter follows the rule of "{generator or function name}_{variable name}". For the
        B-factors and coordinates variables, the variable name follows the rule "{atom name or element name}_{
        Biso, B11, B22, B33, x, y, z}".

        Each parameter in generator is tagged by three tags: the catagory of the paramter ("scale", "delta",
        "lat", "adp", "xyz"), the name of the generator (e. g. "G0") and their union (e. g. "G0_lat"). Each
        parameter in the contribution is tagged by the name of its function if the parameter is named by "{
        function name}_{...}".

        Parameters
        ----------
        scale :
            Whether to add the scale of the generator. Default True.

        delta :
            If "1", add delta1 parameter.
            If "2", add delta2 parameter.
            If None, do nothing.

        lat :
            If "s", constrain the lattice parameters by space group and add the independent variables.
            If "a", add all the lattice paramters.
            If None, do nothing.
            Default "s".

        adp :
            If "a", add all the Biso parameter.
            If the structure is `~pyobjcryst.crystal.Crystal`, this step means add all the Biso of unique atoms.
            If "e", constrain the Biso by elements, add the independent Biso.
            If "s", constrain the B-tensor by space group, add all the independent diagonal terms like B11, B22, B33
            or Biso.
            If None, do nothing.
            Default "a".

        xyz :
            If "s", constrain the coordinates of atoms by space group and add independent coordinates.
            If "a", add all coordinates of atoms.
            If None, do nothing.
            Default "s".

        params :
            If "a", add all the parameters in the equation and characteristic functions like "psize".
            If list of str, add all the parameters whose names are in the list.
            If None, do nothing.
            Default "a".
        """
        initialize(self, scale=scale, delta=delta, lat=lat, adp=adp, xyz=xyz, params=params)

    def optimize(self, tags: tp.List[tp.Union[str, tp.Iterable[str]]], validate: bool = True, verbose: int = 0,
                 **kwargs):
        """First fix all variables and then free the variables one by one and fit the recipe.

        Parameters
        ----------
        tags
            The tags of variables to free. It can be single tag or a tuple of tags.

        validate
            Whether to validate the existences of the tags and variable names before the optimization.

        verbose
            How verbose the fit should be.

        kwargs
            The kwargs of the 'fit'.
        """
        optimize(self, tags=tags, validate=validate, verbose=verbose, **kwargs)

    def visualize(self):
        """View the fit curves. Each FitContribution will be a plot."""
        view_fits(self)


class MyFitResults(FitResults):
    """The augmented fit result interface."""
    pass
