import math
import pathlib
import typing as tp

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import xarray as xr
from diffpy.srfit.fitbase import FitResults
from diffpy.srfit.fitbase.fitresults import initializeRecipe
from diffpy.srfit.fitbase.parameter import Parameter
from diffpy.srfit.fitbase.profile import Profile
from pyobjcryst.crystal import Crystal

import pdffitx.modeling as md


def load_data(filename: str, metadata: dict = None) -> Profile:
    profile = Profile()
    parser = md.MyParser()
    parser.parseFile(filename, metadata)
    profile.loadParsedData(parser)
    return profile


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


class ModelBase:
    """The template for the model class."""

    def __init__(self, equation: str, structures: tp.Dict[str, Crystal],
                 characteristics: tp.Dict[str, tp.Callable]):
        self._equation = equation
        self._structures = structures
        self._characteristics = characteristics
        self._recipe = self._create_recipe()
        self._fit_result = FitResults(self._recipe, update=False)
        self._verbose: int = 1
        self._order: tp.List[tp.Union[str, tp.Iterable[str]]] = []
        self._options: dict = {}

    def parallel(self, ncpu: int):
        fc = self.get_contribution()
        for g in fc.generators.values():
            g.parallel(ncpu)

    def set_xrange(self, xmin: float = None, xmax: float = None, xstep: float = None) -> None:
        profile = self.get_profile()
        profile.setCalculationRange(xmin=xmin, xmax=xmax, dx=xstep)

    def get_xrange(self) -> tp.List:
        return self._xrange

    def set_verbose(self, level: int) -> None:
        self._verbose = level

    def get_verbose(self) -> int:
        return self._verbose

    def set_options(self, **kwargs) -> None:
        self._options = kwargs

    def get_options(self) -> dict:
        return self._options

    def set_order(self, *order: tp.Union[str, tp.Iterable[str]]) -> None:
        order = list(order)
        self._check_order(order)
        self._order = order

    def _check_order(self, order: tp.Any) -> None:
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
        return self._order

    def set_param(self, **kwargs) -> None:
        for name, value in kwargs.items():
            if not hasattr(self._recipe, name):
                raise ValueError("There is no parameter called '{}'".format(name))
        for name, value in kwargs.items():
            var: Parameter = getattr(self._recipe, name)
            var.setValue(value)

    def _create_recipe(self) -> md.MyRecipe:
        raise NotImplemented

    def get_contribution(self) -> md.MyContribution:
        return next(iter(self._recipe.contributions.values()))

    def set_profile(self, profile: Profile) -> None:
        fc: md.MyContribution = self.get_contribution()
        fc.setProfile(profile)

    def optimize(self) -> None:
        md.optimize(self._recipe, self._order, validate=False, verbose=self._verbose, **self._options)

    def update_result(self) -> None:
        return self._fit_result.update()

    def show(self) -> None:
        self._recipe.show()

    def get_result(self) -> dict:
        fr = self._fit_result
        dct = dict()
        n = len(fr.varnames)
        for i in range(n):
            dct[fr.varnames[i]] = fr.varvals[i]
        n = len(fr.fixednames)
        for i in range(n):
            dct[fr.fixednames[i]] = fr.fixedvals[i]
        dct["rw"] = self._fit_result.rw
        return dct

    def get_profile(self) -> Profile:
        fc = self.get_contribution()
        return fc.profile

    def save(self, filepath: str):
        self._fit_result.saveResults(filepath)

    def load(self, filepath: str):
        initializeRecipe(self._recipe, filepath)

    def export_result(self) -> xr.Dataset:
        dct = self.get_result()
        ds = xr.Dataset(dct)
        for name in ds.variables:
            ds[name].attrs["long_name"] = get_symbol(name)
            ds[name].attrs["units"] = get_unit(name)
        return ds

    def export_fits(self) -> xr.Dataset:
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

    def get_structures(self):
        return self._structures

    def export_in_files(self, directory: str, file_prefix: str) -> None:
        directory = pathlib.Path(directory)
        result = self.export_result()
        path = directory.joinpath("{}_result.nc".format(file_prefix))
        result.to_netcdf(path)
        fits = self.export_fits()
        path = directory.joinpath("{}_fits.nc".format(file_prefix))
        fits.to_netcdf(path)
        structures = self.get_structures()
        for name, structure in structures.items():
            path = directory.joinpath("{}_{}.cif".format(file_prefix, name))
            with path.open("w") as f:
                structure.CIFOutput(f)

    def plot(self) -> None:
        md.view_fits(self._recipe)


class MultiPhaseModel(ModelBase):

    def _create_recipe(self) -> md.MyRecipe:
        n = len(self._structures)
        if n != 2:
            raise ValueError("The model needs exactly two structures. Currently, it has {}".format(n))
        pgs = []
        for name, structure in self._structures.items():
            pg = md.PDFGenerator(name)
            pg.setStructure(structure, periodic=True)
            pgs.append(pg)
        fc = md.MyContribution(self.__class__.__name__)
        fc.xname = "x"
        for pg in pgs:
            fc.addProfileGenerator(pg)
        for name, sf in self._characteristics.items():
            fc.registerFunction(sf, name)
        fc.setEquation(self._equation)
        fr = md.MyRecipe()
        fr.clearFitHooks()
        fr.addContribution(fc)
        md.initialize(fr)
        return fr


def plot_fits(fits: xr.Dataset, offset: float = 0., ax: plt.Axes = None, **kwargs) -> None:
    if ax is None:
        ax = plt.gca()
    fits["yobs"].plot.line(ax=ax, marker="o", fillstyle="none", ls="none", **kwargs)
    fits["ycalc"].plot.line(ax=ax, _labels=False, **kwargs)
    diff = fits["y"] - fits["ycalc"]
    shift = offset + fits["y"].min() - diff.max()
    diff += shift
    ax.axhline(shift, ls='--', alpha=0.5, color="black")
    diff.plot.line(ax=ax, _labels=False, **kwargs)
    ax.set_title("")
    return


def plot_fits_along_dim(fits: xr.Dataset, dim: str, num_col: int = 4, offset: float = 0.,
                        figure_config: dict = None, grid_config: dict = None, plot_config: dict = None) -> tp.List[
    plt.Axes]:
    if grid_config is None:
        grid_config = {}
    if plot_config is None:
        plot_config = {}
    if figure_config is None:
        figure_config = {}
    fig: plt.Figure = plt.figure(**figure_config)
    n = len(fits[dim])
    num_row = math.ceil(n / num_col)
    grids = gridspec.GridSpec(num_row, num_col, figure=fig, **grid_config)
    axes = []
    for i, grid in zip(fits[dim], grids):
        fit = fits.isel({dim: i})
        ax = fig.add_subplot(grid)
        axes.append(ax)
        plot_fits(fit, offset, ax=ax, **plot_config)
    return axes
