import inspect
import math
import pathlib
import typing as tp

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from diffpy.srfit.fitbase import FitResults
from diffpy.srfit.fitbase.fitresults import initializeRecipe
from diffpy.srfit.fitbase.parameter import Parameter
from diffpy.srfit.fitbase.profile import Profile
from pyobjcryst.crystal import Crystal

import pdffitx.modeling as md


def get_arg_names(func: tp.Callable) -> tp.List[str]:
    """Get all the names of arguments.

    Parameters
    ----------
    func

    Returns
    -------

    """
    return inspect.getfullargspec(func)[0]


def rename_args(func: tp.Callable, prefix: str, first: str) -> tp.List[str]:
    """Reformat arguments for the characteristic function.

    Parameters
    ----------
    func
    prefix
    first

    Returns
    -------

    """
    names = get_arg_names(func)
    return [first] + [prefix + name for name in names[1:]]


def get_symbol(name: str) -> str:
    """A conventional rule to rename the parameter name to latex version."""
    words = name.split("_")
    if "scale" in words:
        return "scale"
    if "delta2" in words:
        return r"$\delta_2$"
    if "delta1" in words:
        return r"$\delta_1$"
    for word in ("a", "b", "c"):
        if word in words:
            return word
    for word in ("alpha", "beta", "gamma"):
        if word in words:
            return rf"$\{word}$"
    for word in (
        'Uiso', 'U11', 'U12', 'U13', 'U21', 'U22', 'U23', 'U31', 'U32', 'U33',
        'Biso', 'B11', 'B12', 'B13', 'B21', 'B22', 'B23', 'B31', 'B32', 'B33',
    ):
        if word in words:
            return rf"{word[0]}$_{{{word[1:]}}}$({words[1]})"
    for word in ("x", "y", "z"):
        if word in words:
            return rf"{word}({words[1]})"
    for word in ("psize", "psig", "sthick", "thickness", "radius"):
        if word in words:
            return rf"{word}"
    return " ".join(words[1:])


def get_unit(name: str) -> str:
    """A conventional rule to get the unit."""
    words = name.split("_")
    if "scale" in words:
        return ""
    if "delta2" in words:
        return r"Å$^2$"
    if "delta1" in words:
        return r"Å"
    for word in ("a", "b", "c"):
        if word in words:
            return "Å"
    for word in ("alpha", "beta", "gamma"):
        if word in words:
            return "deg"
    for word in (
        'Uiso', 'U11', 'U12', 'U13', 'U21', 'U22', 'U23', 'U31', 'U32', 'U33',
        'Biso', 'B11', 'B12', 'B13', 'B21', 'B22', 'B23', 'B31', 'B32', 'B33',
    ):
        if word in words:
            return "Å$^2$"
    for word in ("x", "y", "z"):
        if word in words:
            return "Å"
    for word in ("psize", "psig", "sthick", "thickness", "radius"):
        if word in words:
            return "Å"
    return ""


def plot_fits(fits: xr.Dataset, offset: float = 0., ax: plt.Axes = None, **kwargs) -> None:
    """Plot the fitted curves."""
    if ax is None:
        ax = plt.gca()
    kwargs.setdefault("xlim", [0, fits["x"][-1].item()])
    kwargs.setdefault("marker", "o")
    kwargs.setdefault("fillstyle", "none")
    kwargs.setdefault("ls", "none")
    fits["yobs"].plot.line(ax=ax, **kwargs)
    ax.plot(fits["x"], fits["ycalc"])
    diff = fits["y"] - fits["ycalc"]
    shift = offset + fits["y"].min() - diff.max()
    diff += shift
    ax.axhline(shift, ls='--', alpha=0.5, color="black")
    ax.plot(fits["x"], diff)
    return


def plot_fits_along_dim(
    fits: xr.Dataset, dim: str, num_row: int = 1, offset: float = 0.,
    figure_config: dict = None, grid_config: dict = None, plot_config: dict = None,
    subplot_config: dict = None
) -> tp.List[plt.Axes]:
    """Plot the fitted curves in multiple panels."""
    if subplot_config is None:
        subplot_config = {}
    n = len(fits[dim])
    num_col = math.ceil(n / num_row)
    if grid_config is None:
        grid_config = {}
    grid_config.setdefault("wspace", 0.25)
    grid_config.setdefault("hspace", 0.25)
    if plot_config is None:
        plot_config = {}
    if figure_config is None:
        figure_config = {}
    figure_config.setdefault("figsize", (4 * num_col, 3 * num_row))
    fig: plt.Figure = plt.figure(**figure_config)
    grids = gridspec.GridSpec(num_row, num_col, figure=fig, **grid_config)
    axes = []
    for i, grid in zip(fits[dim], grids):
        fit = fits.isel({dim: i})
        ax = fig.add_subplot(grid, **subplot_config)
        axes.append(ax)
        plot_fits(fit, offset, ax=ax, **plot_config)
    return axes


class ModelBase:
    """The template for the model class."""

    def __init__(self, recipe: md.MyRecipe):
        self._recipe = recipe
        self._contribution = next(iter(recipe.contributions.values()))
        self._fit_result = FitResults(self._recipe, update=False)
        self._verbose: int = 1
        self._order: tp.List[tp.Union[str, tp.Iterable[str]]] = []
        self._options: dict = {}
        self._fit_state = None

    def parallel(self, ncpu: int) -> None:
        """Parallel computing.

        Parameters
        ----------
        ncpu :
            Number of CPUs.
        """
        fc = self.get_contribution()
        for g in fc.generators.values():
            g.parallel(ncpu)

    def set_xrange(self, start: float = None, end: float = None, step: float = None) -> None:
        """Set fitting range.

        Parameters
        ----------
        start :
            Start of x. x >= start
        end :
            End of x. x <= end
        step :
            Step of x. x[i] - x[i-1] == step

        Returns
        -------
        None
        """
        profile = self.get_profile()
        profile.setCalculationRange(xmin=start, xmax=end, dx=step)

    def set_verbose(self, level: int) -> None:
        """Set verbose level.

        Parameters
        ----------
        level :
            The level used. 0 means quiet.

        Returns
        -------
        None
        """
        self._verbose = level

    def get_verbose(self) -> int:
        """Get verbose level

        Returns
        -------
        Verbose level.
        """
        return self._verbose

    def set_options(self, **kwargs) -> None:
        """Set options for fitting.

        Parameters
        ----------
        kwargs :
            The options for the scipy.optimize.least_squares.

        Returns
        -------
        None
        """
        self._options = kwargs

    def get_options(self) -> dict:
        """Get options for fitting.

        Returns
        -------
        A dictionary of options.
        """
        return self._options

    def set_order(self, *order: tp.Union[str, tp.Iterable[str]]) -> None:
        """Set the order of fitting parameters.

        Parameters
        ----------
        order :
            A list of list or string.

        Returns
        -------
        None

        Examples
        --------
        if order is ["A", ["B", "C"]], "A" will be first refined and "B", "C" will be added after and refined.
        """
        order = list(order)
        self._check_order(order)
        self._order = order

    def _check_order(self, order: tp.Any) -> None:
        """Check the order."""
        tags = set(self._recipe._tagmanager.alltags())
        if isinstance(order, str):
            if not hasattr(self._recipe, order) and order not in tags:
                raise ValueError("'{}' is not in the variable names.".format(order))
        elif isinstance(order, tp.Iterable):
            for x in order:
                self._check_order(x)
        else:
            raise TypeError("'{}' is not allowed.".format(type(order)))

    def get_order(self) -> tp.List[tp.Union[str, tp.Iterable[str]]]:
        """Get the order of the parameters

        Returns
        -------
        A list of parameters.
        """
        return self._order

    def set_value(self, **kwargs) -> None:
        """Set the parameter values.

        Parameters
        ----------
        kwargs :
            In the format of param = value.

        Returns
        -------
        None
        """
        self._check_params(kwargs.keys())
        for name, value in kwargs.items():
            var: Parameter = getattr(self._recipe, name)
            var.setValue(value)

    def get_param(self, name: str) -> Parameter:
        """Get the parameters."""
        if not hasattr(self._recipe, name):
            raise KeyError("No such parameter call '{}' in the recipe.".format(name))
        return getattr(self._recipe, name)

    def set_bound(self, **kwargs) -> None:
        """Set the bound.

        Parameters
        ----------
        kwargs :
            In the form of param = (lb, ub)

        Returns
        -------
        None
        """
        self._check_params(kwargs.keys())
        for name, bound in kwargs.items():
            var: Parameter = getattr(self._recipe, name)
            var.boundRange(*bound)

    def set_rel_bound(self, **kwargs) -> None:
        """Set the bound relatively to current value.

        Parameters
        ----------
        kwargs :
            In the form of param = (lb, ub)

        Returns
        -------
        None
        """
        self._check_params(kwargs.keys())
        for name, bound in kwargs.items():
            var: Parameter = getattr(self._recipe, name)
            var.boundWindow(*bound)

    def _check_params(self, params):
        """Check the parameters."""
        for param in params:
            if not hasattr(self._recipe, param):
                raise KeyError("There is no parameter called '{}'".format(param))

    def _create_recipe(self) -> md.MyRecipe:
        """Place holder for the method to create the recipe."""
        raise NotImplemented

    def get_contribution(self) -> md.MyContribution:
        """Get the first contribution in recipe.

        Returns
        -------
        A FitContribution.
        """
        return self._contribution

    def get_generators(self) -> tp.Dict[str, tp.Callable]:
        """Get the generators in a dictionary."""
        return self.get_contribution().generators

    def calc_phase(self, name: str) -> xr.DataArray:
        """Calculate the data from a generator.

        Parameters
        ----------
        name :
            The name of a generator.

        Returns
        -------
        A xarray.DataArray of calculated y with x as the coordinate.
        """
        gs = self.get_generators()
        p = self.get_profile()
        if name not in gs:
            raise KeyError("There are no generators named '{}'.".format(name))
        y = gs[name](p.x)
        arr = xr.DataArray(y, coords={"x": x}, dims=["x"])
        arr["y"].attrs["standard_name"] = "G"
        arr["y"].attrs["units"] = r"Å$^{-2}$"
        arr["x"].attrs["standard_name"] = "r"
        arr["x"].attrs["units"] = "Å"
        return arr

    def set_profile(self, profile: Profile) -> None:
        """Set the data profile.

        Parameters
        ----------
        profile :
            A data profile.

        Returns
        -------
        None
        """
        fc: md.MyContribution = self.get_contribution()
        fc.setProfile(profile)

    def get_profile(self) -> Profile:
        """Get the data profile."""
        fc = self.get_contribution()
        return fc.profile

    def optimize(self) -> None:
        """Optimize the model. The scipy.optimize.least_squares is used.

        Returns
        -------
        None
        """
        if not self._order:
            raise ValueError("No parameters to refine.")
        md.optimize(self._recipe, self._order, validate=False, verbose=self._verbose, **self._options)
        rw = self.get_rw()
        if self._verbose > 0:
            print("Optimization result: Rw = {:.6f}.".format(rw))

    def get_rw(self) -> float:
        """Calculate Rw value from profile.

        -------
        Rw value.
        """
        profile = self.get_profile()
        y, ycalc = profile.y, profile.ycalc
        return np.sqrt(np.sum((y - ycalc) ** 2) / np.sum(ycalc ** 2))

    def update(self) -> None:
        """Update the result."""
        return self._fit_result.update()

    def show(self) -> None:
        """Show the values of parameters."""
        self._recipe.show()

    def get_result(self) -> dict:
        """Get the result in a dictionary"""
        dct = dict()
        fr = self._fit_result
        n = len(fr.varnames)
        for i in range(n):
            dct[fr.varnames[i]] = fr.varvals[i]
        n = len(fr.fixednames)
        for i in range(n):
            dct[fr.fixednames[i]] = fr.fixedvals[i]
        dct["rw"] = self._fit_result.rw
        return dct

    def save(self, directory: str, file_prefix: str) -> None:
        """Save the model parameters. Must update before save.

        Parameters
        ----------
        directory :
            The directory to export the files.
        file_prefix :
            The prefix of the file name.

        Returns
        -------
        None
        """
        directory = pathlib.Path(directory)
        if not directory.is_dir():
            directory.mkdir(parents=True)
        path = directory.joinpath("{}.txt".format(file_prefix))
        self._fit_result.saveResults(path)

    def load(self, filepath: str) -> None:
        """Load the parameters for the model.

        Parameters
        ----------
        filepath :
            The path to the file or the string of the content or a IOstream.

        Returns
        -------
        None
        """
        initializeRecipe(self._recipe, filepath)

    def export_result(self) -> xr.Dataset:
        """Export the result in a dataset."""
        dct = self.get_result()
        ds = xr.Dataset(dct)
        for name in ds.variables:
            ds[name].attrs["long_name"] = get_symbol(name)
            ds[name].attrs["units"] = get_unit(name)
        ds["rw"].attrs["long_name"] = "$R_w$"
        return ds

    def export_fits(self) -> xr.Dataset:
        """Export the fits in a dataset."""
        profile = self.get_profile()
        ds = xr.Dataset(
            {"y": (["x"], profile.y), "ycalc": (["x"], profile.ycalc), "yobs": (["xobs"], profile.yobs)},
            {"x": (["x"], profile.x), "xobs": (["xobs"], profile.xobs)}
        )
        ds["y"].attrs["standard_name"] = "G"
        ds["y"].attrs["units"] = r"Å$^{-2}$"
        ds["ycalc"].attrs["standard_name"] = "G"
        ds["ycalc"].attrs["units"] = r"Å$^{-2}$"
        ds["yobs"].attrs["standard_name"] = "G"
        ds["yobs"].attrs["units"] = r"Å$^{-2}$"
        ds["x"].attrs["standard_name"] = "r"
        ds["x"].attrs["units"] = "Å"
        ds["xobs"].attrs["standard_name"] = "r"
        ds["xobs"].attrs["units"] = "Å"
        return ds

    def save_result(self, directory: str, file_prefix: str) -> None:
        """Save the fitting result.

        Parameters
        ----------
        directory :
            The directory to export the files.
        file_prefix :
            The prefix of the file name.

        Returns
        -------
        None
        """
        directory = pathlib.Path(directory)
        if not directory.is_dir():
            directory.mkdir(parents=True)
        result = self.export_result()
        path = directory.joinpath("{}_result.nc".format(file_prefix))
        result.to_netcdf(path)

    def save_fits(self, directory: str, file_prefix: str) -> None:
        """Save the fitted curves.

        Parameters
        ----------
        directory :
            The directory to export the files.
        file_prefix :
            The prefix of the file name.

        Returns
        -------
        None
        """
        directory = pathlib.Path(directory)
        if not directory.is_dir():
            directory.mkdir(parents=True)
        fits = self.export_fits()
        path = directory.joinpath("{}_fits.nc".format(file_prefix))
        fits.to_netcdf(path)

    def save_all(self, directory: str, file_prefix: str) -> None:
        """Save the results, fits and structures in a directory.

        Parameters
        ----------
        directory :
            The directory to export the files.
        file_prefix :
            The prefix of the file name.

        Returns
        -------
        None
        """
        self.save(directory, file_prefix)
        self.save_result(directory, file_prefix)
        self.save_fits(directory, file_prefix)

    def plot(self, **kwargs) -> None:
        """View the fitted curves.

        Returns
        -------
        None
        """
        fits = self.export_fits()
        plot_fits(fits, **kwargs)

    def get_recipe(self) -> md.MyRecipe:
        """Get the recipe.

        Returns
        -------
        The recipe in the model.
        """
        return self._recipe


class MultiPhaseModel(ModelBase):
    """The model for multi-phase fitting of PDFs."""

    def __init__(self, equation: str, structures: tp.Dict[str, Crystal] = None,
                 characteristics: tp.Dict[str, tp.Callable] = None, **kwargs):
        if structures is None:
            structures = {}
        if characteristics is None:
            characteristics = {}
        self._equation = equation
        self._structures = structures
        self._characteristics = characteristics
        self._init_mode = kwargs
        recipe = self._create_recipe()
        super(MultiPhaseModel, self).__init__(recipe)

    def _create_recipe(self) -> md.MyRecipe:
        pgs = []
        for name, structure in self._structures.items():
            pg = md.PDFGenerator(name)
            pg.setStructure(structure, periodic=True)
            pgs.append(pg)
        fc = md.MyContribution(self.__class__.__name__)
        fc.setProfile(Profile())
        for pg in pgs:
            fc.addProfileGenerator(pg)
        for name, sf in self._characteristics.items():
            argnames = rename_args(sf, "{}_".format(name), fc.xname)
            fc.registerFunction(sf, name, argnames)
        fc.setEquation(self._equation)
        fr = md.MyRecipe()
        fr.clearFitHooks()
        fr.addContribution(fc)
        md.initialize(fr, **self._init_mode)
        return fr

    def get_equation(self) -> str:
        """Get the equation."""
        return self.get_contribution().getEquation()

    def get_structures(self) -> tp.Dict[str, Crystal]:
        """Get the structures in a directory."""
        return self._structures

    def get_characteristics(self) -> tp.Dict[str, tp.Callable]:
        """Get all characteristic functions."""
        return self._characteristics

    def save_all(self, directory: str, file_prefix: str) -> None:
        """Export the results, fits and structures in a directory."""
        super(MultiPhaseModel, self).save_all(directory, file_prefix)
        self.save_structures(directory, file_prefix)

    def save_structures(self, directory: str, file_prefix: str) -> None:
        """Save the structures."""
        directory = pathlib.Path(directory)
        if not directory.is_dir():
            directory.mkdir(parents=True)
        structures = self.get_structures()
        for name, structure in structures.items():
            path = directory.joinpath("{}_{}.cif".format(file_prefix, name))
            with path.open("w") as f:
                structure.CIFOutput(f)
