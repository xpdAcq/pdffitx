import dataclasses as dc
import typing
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import ipywidgets.widgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.nonparametric.smoothers_lowess as smoothers_lowess
import xarray as xr
from diffpy.pdfgetx import PDFConfig, PDFGetter, Transformation
from diffpy.pdfgetx.pdfconfig import PDFConfigError
from ipywidgets import interact
from pyFAI.azimuthalIntegrator import AzimuthalIntegrator

from pdffitx.vend import mask_img, generate_binner


class MyPDFConfig(PDFConfig):
    """PDFConfig for the lowess smoothing.

    Attributes
    ----------
    qcutoff :
        The Q > qcutoff region will be LOWESS smoothed.
    lowessf :
        The frac parameter used in LOWESS smoothing. The larger it is, the smoother it will be.
    """

    def __init__(self):
        super(MyPDFConfig, self).__init__()
        self.qcutoff = 24.0
        self.lowessf = 0.04
        self.endzero = True
        self.dataformat = "QA"
        self.qmin = 0.
        self.qmaxinst = 24.0
        self.qmax = 22.0


def smooth(xin: np.ndarray, yin: np.ndarray, xcutoff: float, lowessf: float, endzero: bool) -> typing.Tuple[
    np.ndarray, np.ndarray]:
    """Smooth the input data in region x >= xcutoff using lowessf parameter. If endzero True, terminate the data to the last zero point."""
    xout, yout = xin.copy(), yin.copy()
    cutoff = np.searchsorted(xin, xcutoff) + 1
    if cutoff < xin.shape[0]:
        xout[cutoff:], yout[cutoff:] = smoothers_lowess.lowess(yin[cutoff:], xin[cutoff:], frac=lowessf).T
    if endzero:
        # first element with a different sign
        ind = np.argmin(np.sign(yout[-1] * yout[::-1]))
        xout, yout = xout[:xout.shape[0] - ind], yout[:yout.shape[0] - ind]
    return xout, yout


class LowessTransform(Transformation):
    """The transformation doing the LOWESS smoothing on F(Q)."""

    summary = "LOWESS smoothing"
    outputtype = "lsfq"
    xinlabel = ""
    yinlabel = ""
    xoutlabel = ""
    youtlabel = ""

    xin = None
    yin = None
    xout = None
    yout = None

    def __init__(self, config: MyPDFConfig):
        super(LowessTransform, self).__init__(config)

    def checkConfig(self):
        if not isinstance(self.config, MyPDFConfig):
            raise PDFConfigError("The config for LowessTransform must be LowessPDFConfig.")

    def transform(self):
        self.xout, self.yout = smooth(self.xin, self.yin, self.config.qcutoff, self.config.lowessf,
                                      self.config.endzero)


class MyPDFGetter(PDFGetter):
    """The PDFGetter with LOWESS smoothing of F(Q) included."""

    def __init__(self, config: MyPDFConfig):
        super(MyPDFGetter, self).__init__(config)
        self.transformations.insert(-1, LowessTransform(config))


@dataclass
class MaskConfig:
    edge: int = 30
    lower_thresh: float = 0.0
    upper_thresh: float = None
    alpha: float = 2.
    auto_type: str = "median"
    tmsk: np.ndarray = None


@dataclass
class IntegrationConfig:
    npt: int = 3001
    correctSolidAngle: bool = False
    dummy: float = 0.
    unit: str = "q_A^-1"
    safe: bool = False
    polarization_factor: float = 0.99
    method: typing.Tuple[str, str, str] = ("bbox", "csr", "cython")


@dataclass
class BackgroundConfig:
    scale: float = 1.


@dataclass
class LabelConfig:
    Q: str = "Q"
    F: str = "F"
    I: str = "I"
    r: str = "r"
    G: str = "G"
    A: str = "Å"
    QU: str = "Å$^{-1}$"
    IU: str = "A. U."
    rU: str = "Å"
    GU: str = "Å$^{-2}$"
    FU: str = "Å$^{-1}$"


@dataclass
class ReductionConfig:
    geometry: AzimuthalIntegrator = AzimuthalIntegrator()
    mask: MaskConfig = MaskConfig()
    integration: IntegrationConfig = IntegrationConfig()
    background: BackgroundConfig = BackgroundConfig()
    pdf: MyPDFConfig = MyPDFConfig()
    label: LabelConfig = LabelConfig()


class ReductionCalculator:

    def __init__(self, config: ReductionConfig):
        self.config: ReductionConfig = config
        self.dataset: xr.Dataset = xr.Dataset()
        self.dark_dataset: xr.Dataset = xr.Dataset()
        self.bkg_dataset: xr.Dataset = xr.Dataset()
        self.executor = ThreadPoolExecutor(max_workers=24)

    def set_dataset(self, dataset: xr.Dataset) -> None:
        self.dataset = dataset
        return

    def set_dark_dataset(self, dataset: xr.Dataset) -> None:
        self.dark_dataset = dataset
        return

    def set_bkg_dataset(self, dataset: xr.Dataset) -> None:
        self.bkg_dataset = dataset
        return

    @staticmethod
    def _average(
            ds: xr.Dataset,
            image_name: str,
            along: typing.Sequence[str]
    ):
        n = len(along)
        averaged = xr.apply_ufunc(
            np.mean,
            ds[image_name],
            input_core_dims=[along],
            exclude_dims=set(along),
            kwargs={"axis": tuple(range(-n, 0))}
        )
        ds = ds.assign({image_name: averaged})
        return ds

    def average_dark(
            self,
            image_name: str,
            along: typing.Sequence[str] = ("time", "dim_0")
    ):
        self.dark_dataset = self._average(
            self.dark_dataset,
            image_name,
            along
        )
        return

    def average_bkg(
            self,
            image_name: str,
            along: typing.Sequence[str] = ("time", "dim_0")
    ):
        self.bkg_dataset = self._average(
            self.bkg_dataset,
            image_name,
            along
        )

    def average(
            self,
            image_name: str,
            along: typing.Sequence[str] = ("dim_0",)
    ):
        self.dataset = self._average(
            self.dataset,
            image_name,
            along
        )
        return

    def _dark_subtract(
            self,
            ds,
            image_name: str,
            image_dims: typing.Sequence[str]
    ):
        dark_ds = self.dark_dataset
        corrected = xr.apply_ufunc(
            np.subtract,
            ds[image_name],
            dark_ds[image_name].values,
            input_core_dims=[image_dims, image_dims],
            output_core_dims=[image_dims]
        )
        ds = ds.assign({image_name: corrected})
        return ds

    def dark_subtract(
            self,
            image_name: str,
            image_dims: typing.Sequence[str] = ("dim_1", "dim_2")
    ):
        self.dataset = self._dark_subtract(
            self.dataset,
            image_name,
            image_dims
        )
        return

    def dark_subtract_bkg(
            self,
            image_name: str,
            image_dims: typing.Sequence[str] = ("dim_1", "dim_2")
    ):
        self.bkg_dataset = self._dark_subtract(
            self.bkg_dataset,
            image_name,
            image_dims
        )

    def mask(
            self,
            image_name: str,
            image_dims: typing.Sequence[str] = ("dim_1", "dim_2")
    ) -> None:
        ds = self.dataset
        ai = self.config.geometry
        mc = self.config.mask
        kwargs = dc.asdict(mc)
        shape = [ds.dims[d] for d in image_dims]
        kwargs["binner"] = generate_binner(ai, shape)
        mask = xr.apply_ufunc(
            mask_img,
            ds[image_name],
            kwargs=kwargs,
            input_core_dims=[image_dims],
            output_core_dims=[image_dims],
            vectorize=True
        )
        ds = ds.assign({image_name: ds[image_name] * mask})
        self.dataset = ds
        return

    def _integrate(
            self,
            ds: xr.Dataset,
            image_name: str,
            image_dims: typing.Tuple[str, str],
            chi_name: str,
            q_name: str
    ) -> xr.Dataset:
        ai = self.config.geometry
        exe = self.executor
        ic = self.config.integration
        images = ds[image_name]
        other_dims = tuple(set(images.dims) - set(image_dims))
        images.transpose(*other_dims, *image_dims)
        kwargs = dc.asdict(ic)
        images_data = images.data
        if len(other_dims) > 0:
            res = np.asarray(
                list(exe.map(lambda img: ai.integrate1d(img, **kwargs), images_data))
            )
            q = res[0, 0, :]
            i = res[:, 1, :]
        else:
            res = ai.integrate1d(images_data, **kwargs)
            q = res[0]
            i = res[1]
        dims = other_dims + (q_name,)
        ds = ds.assign_coords({q_name: q})
        ds = ds.assign({chi_name: (dims, i)})
        return ds

    def integrate(
            self,
            image_name: str,
            image_dims: typing.Tuple[str, str] = ("dim_1", "dim_2"),
            chi_name: str = "I",
            q_name: str = "Q"
    ):
        self.dataset = self._integrate(
            self.dataset,
            image_name,
            image_dims,
            chi_name,
            q_name
        )
        return

    def integrate_bkg(
            self,
            image_name: str,
            image_dims: typing.Tuple[str, str] = ("dim_1", "dim_2"),
            chi_name: str = "I",
            q_name: str = "Q"
    ):
        self.bkg_dataset = self._integrate(
            self.bkg_dataset,
            image_name,
            image_dims,
            chi_name,
            q_name
        )

    def bkg_subtract(
            self,
            chi_name: str = "I",
            q_name: str = "Q"
    ):
        scale = self.config.background.scale
        ds = self.dataset
        bkg_ds = self.bkg_dataset
        subtracted = xr.apply_ufunc(
            lambda x, y: np.subtract(x, scale * y),
            ds[chi_name],
            bkg_ds[chi_name],
            input_core_dims=[[q_name], [q_name]],
            output_core_dims=[[q_name]]
        )
        self.dataset = self.dataset.assign(
            {chi_name: subtracted}
        )
        return

    def get_G(
            self,
            chi_name: str = "I",
            q_name: str = "Q",
            g_name: str = "G",
            r_name: str = "r"
    ):
        ds = self.dataset
        label = self.config.label
        x = ds[q_name].data
        mpg = MyPDFGetter(self.config.pdf)
        g = xr.apply_ufunc(
            lambda y: mpg.__call__(x, y)[1],
            ds[chi_name],
            input_core_dims=[[q_name]],
            output_core_dims=[[r_name]],
            exclude_dims={q_name},
            vectorize=True
        )
        r = xr.DataArray(mpg.gr[0], dims=[r_name], attrs={"units": label.rU, "standard_name": label.r})
        g.attrs.update({"units": label.GU, "standard_name": label.G})
        ds = ds.assign_coords({r_name: r})
        ds = ds.assign({g_name: g})
        self.dataset = ds
        return

    def interact_fq(
            self,
            index: int = 0,
            chi_name: str = "I",
            q_name: str = "Q"
    ):
        i = self.dataset[chi_name][index]
        q = i[q_name]
        mpg = MyPDFGetter(self.config.pdf)
        config = mpg.config
        pdf_config = self.config.pdf
        label = self.config.label

        def func(
                qmin,
                qmax,
                qmaxinst,
                lowessf,
                qcutoff,
                endzero
        ):
            config.qmin = qmin
            config.qmax = qmax
            config.qmaxinst = qmaxinst
            config.lowessf = lowessf
            config.qcutoff = qcutoff
            config.endzero = endzero
            mpg.__call__(q, i)
            q1, f1 = mpg.t[-3].xout, mpg.t[-3].yout
            q2, f2 = mpg.t[-2].xout, mpg.t[-2].yout
            ax: plt.Axes = plt.subplots()[1]
            ax.plot(q1, f1)
            ax.plot(q2, f2)
            ax.set_xlabel("{} [{}]".format(label.Q, label.QU))
            ax.set_ylabel("{} [{}]".format(label.F, label.FU))
            plt.pause(0.1)

        return interact(
            func,
            qmin=widgets.FloatSlider(pdf_config.qmin, vmin=0., vmax=2.),
            qmax=widgets.FloatSlider(pdf_config.qmax, vmin=10., vmax=30.0),
            qmaxinst=widgets.FloatSlider(pdf_config.qmaxinst, vmin=20.0, vmax=30.0),
            lowessf=widgets.FloatSlider(pdf_config.lowessf, vmin=0.0, vmax=0.1),
            qcutoff=widgets.FloatSlider(pdf_config.qcutoff, vmin=0.0, vmax=30.0),
            endzero=widgets.Checkbox(pdf_config.endzero)
        )

    def get_I(
            self,
            image_name: str,
            chi_name: str = "I",
            q_name: str = "Q",
            avg_along: typing.Sequence[str] = ("dim_0",),
            dark_avg_along: typing.Sequence[str] = ("time", "dim_0"),
            bkg_avg_along: typing.Sequence[str] = ("time", "dim_0"),
            image_dims: typing.Tuple[str, str] = ("dim_1", "dim_2"),
            drop_image: bool = True
    ):
        self.average(image_name, avg_along)
        if image_name in self.dark_dataset:
            self.average_dark(image_name, dark_avg_along)
            self.dark_subtract(image_name, image_dims)
        self.mask(image_name, image_dims)
        self.integrate(image_name, image_dims, chi_name, q_name)
        if image_name in self.bkg_dataset:
            self.average_bkg(image_name, bkg_avg_along)
            self.dark_subtract_bkg(image_name, image_dims)
            self.integrate_bkg(image_name, image_dims, chi_name, q_name)
            self.bkg_subtract(chi_name)
        ds = self.dataset
        if drop_image:
            ds = ds.drop_vars(image_name)
        label = self.config.label
        ds[chi_name].attrs.update({"units": label.IU, "standard_name": label.I})
        ds[q_name].attrs.update({"units": label.QU, "standard_name": label.Q})
        self.dataset = ds
        return

    def reset_dims(
            self,
            dim2dims: typing.Dict[str, typing.List[str]],
    ):
        self.dataset = reset_dims(self.dataset, dim2dims)
        return


def reset_dims(
        ds: xr.Dataset,
        dim2dims: typing.Dict[str, typing.List[str]]
) -> xr.Dataset:
    # set new dims
    old_dims = list(dim2dims.keys())
    ds = ds.reset_index(old_dims)
    ds = ds.set_index(dim2dims)
    ds = ds.unstack()
    # rename new dims
    replaced_dims = {}
    for old_dim, new_dims in dim2dims.items():
        if isinstance(new_dims, str):
            replaced_dims[old_dim] = new_dims
        elif len(new_dims) == 1:
            replaced_dims[old_dim] = new_dims[0]
    if len(replaced_dims) > 0:
        ds = ds.rename_dims(replaced_dims)
        ds = ds.rename_vars(replaced_dims)
    # rename old dims (coords now)
    rule = {"{}_".format(old_dim): old_dim for old_dim in old_dims}
    ds = ds.rename_vars(rule)
    return ds
