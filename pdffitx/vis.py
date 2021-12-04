import typing
from collections import ChainMap

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from xarray.plot import FacetGrid


def _get_offset(arr: np.ndarray, delta: float = 0.) -> np.ndarray:
    maxs = np.max(arr, axis=1)
    mins = np.min(arr, axis=1)
    offsets = mins[:-1] - maxs[1:] + delta
    offsets = np.insert(offsets, 0, 0)
    offsets = np.cumsum(offsets)
    return offsets


def get_offset(da: xr.DataArray, delta: float = 0.) -> xr.DataArray:
    return xr.apply_ufunc(
        lambda x: _get_offset(x, delta),
        da,
        input_core_dims=[da.dims],
        output_core_dims=[[da.dims[0]]]
    )


def plot_waterfall(
        ds: xr.Dataset,
        y: str,
        *,
        offset: typing.Sequence[float] = None,
        lab: str = None,
        delta: float = 0.,
        ax: plt.Axes = None,
        line_kws: dict = None,
        hline: bool = True,
        axhline_kws: dict = None,
        lab_xys: typing.List[typing.Tuple[float, float]] = None
) -> plt.Axes:
    """Plot the waterfall lines.

    Parameters
    ----------
    ds :
        The dataset.
    y :
        The name of the y values. It should be a 2d array.
    offset :
        The zero ground line levels.
    lab :
        The name of the labels.
    delta :
        Add more or less space between the curves. Unit is the unit of the data.
    ax :
        The Axes to plot. If None, create a new figure.
    line_kws :
        The key words for the line plots.
    hline :
        Whether to plot the zero baselines or not.
    axhline_kws :
        The key words for the axhline plots.
    lab_xys :
        The lilst of xy values for the labels. If None, it will be calculated in best effort.

    Returns
    -------
    The Axes.
    """
    if ax is None:
        _, ax = plt.subplots()
    if line_kws is None:
        line_kws = {}
    if axhline_kws is None:
        axhline_kws = {}
    hue = ds[y].dims[0]
    x = ds[y].dims[1]
    # set the default color and line style
    axhline_kws = dict(
        ChainMap(
            axhline_kws,
            {"color": "gray", "ls": "--"}
        )
    )
    # turn off legend if lab is specified
    if lab:
        line_kws["add_legend"] = False
    # get the offsets for the lines
    offset = xr.DataArray(offset, dims=[ds[y].dims[0]]) if offset else get_offset(ds[y], delta)
    # shift the lines
    ydata = ds[y] + offset
    ydata.attrs = ds[y].attrs
    # iterate along the hue dimension to add baselines
    if hline:
        n = offset.shape[0]
        for i in range(n):
            ax.axhline(offset[i].item(), **axhline_kws)
    # plot the lines
    lines = ydata.plot.line(**line_kws, hue=hue, ax=ax)
    # add labels
    if lab:
        if lab_xys is None:
            ymax = ax.get_ylim()[1]
            ybs = np.insert(offset.values, 0, [ymax])
            xc = ds[x].max().item() * 0.75
            ycs = 0.3 * ybs[:-1] + 0.7 * ybs[1:]
            lab_xys = [(xc, yc) for yc in ycs]
        for text, xy, line in zip(ds[lab].values, lab_xys, lines):
            ax.annotate(str(text), xy, va="center", ha="center", color=line.get_color())
    # remove the title
    ax.set_title("")
    return ax


def plot_grids(
        ds: xr.Dataset,
        facet_kws: dict = None,
        plot_func: typing.Callable = xr.plot.line,
        plot_kws: dict = None
) -> FacetGrid:
    """Plot all variables in a dataset in a grid.

    Parameters
    ----------
    ds :
        The dataset.
    facet_kws :
        The key words for xarray.plot.FacetGrid.
    plot_func :
        The function used to plot.
    plot_kws :
        The key words for the xarray.DataArray.plot.

    Returns
    -------
    The FacetGrid object.
    """
    if facet_kws is None:
        facet_kws = {}
    if plot_kws is None:
        plot_kws = {}
    # add default setting
    facet_kws = dict(
        ChainMap(
            facet_kws,
            {"sharex": False, "sharey": False}
        )
    )
    # get all the variable names
    names = [name for name in ds]
    n = len(names)
    # add it as an additional dimension
    dims = dict(ds.dims)
    dims["variable"] = n
    # create grid
    fg = FacetGrid(
        xr.DataArray(
            np.zeros(tuple(dims.values())),
            coords=ds.coords,
            dims=list(dims.keys())
        ),
        col="variable",
        **facet_kws
    )
    # get the axes
    axes: typing.Sequence[plt.Axes] = fg.axes.flatten()
    for i in range(n):
        plot_func(ds[names[i]], ax=axes[i], **plot_kws)
    m = len(axes)
    for i in range(n, m):
        axes[i].axis("off")
    # tight layout
    fg.fig.tight_layout()
    return fg


def _get_label(da: xr.DataArray):
    ss = ("long_name", "standard_name", "short_name", "name")
    name = da.name
    for s in ss:
        if s in da.attrs:
            name = da.attrs[s]
            break
    units = da.attrs.get("units")
    units = "[{}]".format(units) if units else ""
    return "{} {}".format(name, units)


def gridplot_vars(
        ds: xr.Dataset,
        plot_func: typing.Callable[[xr.Dataset, plt.Axes, typing.Any], typing.Any],
        *,
        facet_kws: dict = None,
        plot_kws: dict = None,
        set_titles: bool = True,
        set_labels: bool = False
) -> FacetGrid:
    if facet_kws is None:
        facet_kws = {}
    if plot_kws is None:
        plot_kws = {}
    # add default setting
    facet_kws = dict(
        ChainMap(
            facet_kws,
            {"sharex": False, "sharey": False}
        )
    )
    # get all the variable names
    names = [name for name in ds]
    n = len(names)
    # add it as an additional dimension
    dims = dict(ds.dims)
    dims["variable"] = n
    # create grid
    fg = FacetGrid(
        xr.DataArray(
            np.zeros(tuple(dims.values())),
            coords={k: v for k, v in ds.coords.items() if k in dims},
            dims=list(dims.keys())
        ),
        col="variable",
        **facet_kws
    )
    # get the axes
    axes: typing.Sequence[plt.Axes] = fg.axes.flatten()
    for i in range(n):
        plot_func(ds[names[i]], axes[i], **plot_kws)
    m = len(axes)
    for i in range(n, m):
        axes[i].axis("off")
    # set titles and labels
    labels = [_get_label(ds[name]) for name in ds]
    if set_titles:
        for i in range(n):
            axes[i].set_title(labels[i])
    if set_labels:
        for i in range(n):
            axes[i].set_ylabel(labels[i])
    # tight layout
    fg.fig.tight_layout()
    return fg


def plot_bar(ds: xr.Dataset, ax: plt.Axes, **kwargs) -> None:
    ds.to_dataframe().plot.bar(ax=ax, **kwargs)
    return


def barplot_vars(
        ds: xr.Dataset,
        *,
        facet_kws: dict = None,
        plot_kws: dict = None,
        set_titles: bool = False,
        set_labels: bool = False
) -> FacetGrid:
    # no legend
    plot_kws.setdefault("legend", False)
    return gridplot_vars(
        ds, plot_bar, facet_kws=facet_kws, plot_kws=plot_kws, set_titles=set_titles, set_labels=set_labels
    )


def lineplot_vars(
        ds: xr.Dataset,
        *,
        facet_kws: dict = None,
        plot_kws: dict = None,
        set_titles: bool = False,
        set_labels: bool = False
) -> FacetGrid:
    return gridplot_vars(
        ds, xr.plot.line, facet_kws=facet_kws, plot_kws=plot_kws, set_titles=set_titles, set_labels=set_labels
    )


def imshow_vars(
        ds: xr.Dataset,
        *,
        facet_kws: dict = None,
        plot_kws: dict = None,
        set_titles: bool = False,
        set_labels: bool = False
) -> FacetGrid:
    return gridplot_vars(
        ds, xr.plot.imshow, facet_kws=facet_kws, plot_kws=plot_kws, set_titles=set_titles, set_labels=set_labels
    )


def xrplot_vars(
        ds: xr.Dataset,
        *,
        facet_kws: dict = None,
        plot_kws: dict = None,
        set_titles: bool = False,
        set_labels: bool = False
) -> FacetGrid:
    return gridplot_vars(
        ds, xr.plot.imshow, facet_kws=facet_kws, plot_kws=plot_kws, set_titles=set_titles, set_labels=set_labels
    )
