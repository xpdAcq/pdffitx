"""Create a recipe."""
import inspect
import typing as tp

from diffpy.structure import Structure
from pyobjcryst.crystal import Crystal

from pdffitx.core import MyRecipe, MyContribution
from .fitfuncs import make_contribution
from .fitobjs import FunConfig, GenConfig, ConConfig, MyParser

S = tp.Union[Crystal, Structure]


def create(
    name: str,
    data: MyParser,
    arange: tp.Tuple[float, float, float],
    equation: str,
    functions: tp.Dict[str, tp.Callable],
    structures: tp.Dict[str, S],
    ncpu: int = None
) -> MyRecipe:
    """Create a single-contribution recipe without any variables inside.

    Parameters
    ----------
    name :
        The name of the contribution.

    data :
        The parser that contains the information of the

    arange :
        The rmin, rmax, rstep (inclusive).

    equation :
        The equation of the contribution.

    functions :
        The keys are function names in the equation and the values are function objects.

    structures :
        The keys are structure name in the equation and the values are structure object.

    ncpu :
        The number of cpu used in parallel computing. If None, no parallel. Default None.

    Returns
    -------
    recipe :
        A single-contribution recipe without any variables inside.
    """
    con = create_con(name, data, arange, equation, functions, structures, ncpu)
    recipe = MyRecipe()
    recipe.addContribution(con)
    recipe.clearFitHooks()
    return recipe


def create_con(
    name: str,
    data: MyParser,
    arange: tp.Tuple[float, float, float],
    equation: str,
    functions: tp.Dict[str, tp.Callable],
    structures: tp.Dict[str, S],
    ncpu: int = None
) -> MyContribution:
    """Make a contribution."""
    genconfigs = [
        GenConfig(
            name=n, structure=s, ncpu=ncpu
        )
        for n, s in structures.items()
    ]
    funconfigs = [
        FunConfig(
            name=n, func=f, argnames=add_prefix(f, n)
        )
        for n, f in functions.items()
    ]
    conconfig = ConConfig(
        name=name, eq=equation, parser=data, fit_range=arange,
        genconfigs=genconfigs, funconfigs=funconfigs
    )
    return make_contribution(conconfig)


def add_prefix(func: tp.Callable, prefix: str, xname: str = "r") -> tp.List[str]:
    """Add the suffix to the argument names starting at the second the argument. Return the names"""
    args = inspect.getfullargspec(func).args
    return list(
        map(
            lambda arg: '{}_{}'.format(prefix, arg) if arg != xname else arg,
            args
        )
    )
