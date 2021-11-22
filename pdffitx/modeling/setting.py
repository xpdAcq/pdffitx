"""Set hyper parameters in the recipe."""
import typing as tp

from diffpy.srfit.fitbase.parameter import Parameter
from numpy import ndarray

from pdffitx.modeling.core import MyRecipe


def set_range(
        recipe: MyRecipe,
        rmin: tp.Tuple[str, float] = None,
        rmax: tp.Tuple[str, float] = None,
        rstep: tp.Tuple[str, float] = None
):
    """Set fitting range of the single contribution in the recipe.

    Parameters
    ----------
    recipe :
        The recipe. It has only one contribution with profile.

    rmin :
        The minimum value of fitting range (inclusive). If None, keep original value. If "obs", use data value.

    rmax :
        The maximum value of fitting range (inclusive). If None, keep original value. If "obs", use data value.

    rstep :
        The step of fitting range. If None, keep original value. If "obs", use data value.
    """
    con = next(iter(recipe.contributions.values()))
    con.profile.setCalculationRange(xmin=rmin, xmax=rmax, dx=rstep)
    return


def get_range(recipe: MyRecipe) -> ndarray:
    """Get the fitting range of the single contribution in recipe.

    Parameters
    ----------
    recipe :
        The recipe with a single contribution.

    Returns
    -------
    x :
        The array of the x values in the fitting.
    """
    con = next(iter(recipe.contributions.values()))
    return con.profile.x


def get_variable(recipe: MyRecipe, name: str, ignore: bool = False) -> Parameter:
    """Get a fitting parameter from the recipe."""
    variable = getattr(recipe, name, None)
    if not variable:
        if not ignore:
            raise ValueError("Recipe doesn't have parameter '{}'.".format(name))
    return variable


def set_values(recipe: MyRecipe, values: tp.Dict[str, float], ignore: bool = False) -> MyRecipe:
    """Set the values of fitting parameters in the recipe.

    Parameters
    ----------
    recipe :
        The recipe.

    values :
        The mapping from name of the parameter to its set value.

    ignore :
        If True, ignore the parameter when it is not found in the recipe.

    Returns
    -------
    recipe :
        The input recipe with operation done in place.
    """
    for name, value in values.items():
        variable = get_variable(recipe, name, ignore=ignore)
        if variable:
            variable.setValue(value)
    return recipe


def get_value_dct(recipe: MyRecipe) -> tp.Dict[str, float]:
    """Get the values in the recipe in a dictionary."""
    return dict(zip(recipe.getNames(), recipe.getValues()))


def get_values(recipe: MyRecipe, names: tp.Iterable[str]) -> tp.List[tp.Union[float, None]]:
    """Get the values of the fitting parameters in the recipe.

    Parameters
    ----------
    recipe :
        The recipe.

    names :
        The names of parameters.

    Returns
    -------
    values :
        A list of values in the same order of names. If a value is None, the name is not in the recipe.
    """
    dct = get_value_dct(recipe)
    return list(map(dct.get, names))


def bound_ranges(
        recipe: MyRecipe, bounds: tp.Dict[str, tp.Union[tp.Tuple, tp.Dict]],
        ignore: bool = False, ratio: bool = False
):
    """Bound the variables in the recipe by in (lower bound, upper bound).

    Parameters
    ----------
    recipe :
        The recipe.

    bounds :
        A tuple of lower bound and upper bound or a dictionary with keys "lb", "ub".

    ignore :
        If True, ignore the parameter when it is not found in the recipe.

    ratio :
        If True, the bound is a ratio. The real bound will be the value of the variable * the value of bound.
        e. g. bound (0.1, 1.1) means that the lower bound is 10% and the upper bound is 110% of the initial
        value of the variable.
    """
    for name, bound in bounds.items():
        variable = get_variable(recipe, name, ignore=ignore)
        if variable:
            bound_range(variable, bound, ratio=ratio)
    return


def bound_range(variable: Parameter, bound: tp.Union[tp.Tuple, tp.Dict], ratio: bool = False) -> Parameter:
    """Bound variable by range."""
    value = variable.getValue()
    if isinstance(bound, dict):
        if ratio:
            for k, r in bound.items():
                bound[k] = value * r
        variable.boundRange(**bound)
    else:
        if ratio:
            bound = tuple((r * value for r in bound))
        variable.boundRange(*bound)
    return variable


def bound_windows(
        recipe: MyRecipe, bounds: tp.Dict[str, tp.Union[float, tp.Tuple, tp.Dict]],
        ignore: bool = False, ratio: bool = False
):
    """Bound the variables in the recipe by (variable - lower bound, variable + upper bound).


    Parameters
    ----------
    recipe :
        The recipe.

    bounds :
        A tuple of lower bound and upper bound or a dictionary with keys "lr", "lr".

    ignore :
        If True, ignore the parameter when it is not found in the recipe.

    ratio :
        If True, the bound is a ratio. The real bound will be the value of the variable * the value of bound.
        e. g. bound (0.1, 0.2) means that the lower bound is 90% and the upper bound is 120% of the initial
        value of the variable.
    """
    for name, bound in bounds.items():
        variable = get_variable(recipe, name, ignore=ignore)
        if variable:
            bound_window(variable, bound, ratio=ratio)
    return


def bound_window(
        variable: Parameter, bound: tp.Union[float, tp.Tuple, tp.Dict], ratio: bool = False
) -> Parameter:
    """Bound variable by window."""
    value = variable.getValue()
    if isinstance(bound, dict):
        if ratio:
            for k, r in bound.items():
                bound[k] = value * r
        variable.boundWindow(**bound)
    elif isinstance(bound, float):
        if ratio:
            bound = bound * value
        variable.boundWindow(bound)
    else:
        if ratio:
            bound = tuple((r * value for r in bound))
        variable.boundWindow(*bound)
    return variable


def get_bound_dct(recipe: MyRecipe) -> tp.Dict[str, tp.List[float]]:
    """Get the bounds in the recipe in a dictionary."""
    return dict(zip(recipe.getNames(), recipe.getBounds()))


def get_bounds(recipe: MyRecipe, names: tp.Iterable[str]) -> tp.List[tp.List[float]]:
    """Get the bounds in a recipe.

    Parameters
    ----------
    recipe :
        The recipe.

    names :
        The names of the variables.

    Returns
    -------
    bounds :
        The bounds, each is a list of lower bound and upper bound. If name is not in recipe, the bound is None.
    """
    dct = get_bound_dct(recipe)
    return list(map(dct.get, names))
