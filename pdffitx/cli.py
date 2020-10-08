"""The functions used in the command line interface. The input and output are all files."""
import typing as tp
from pathlib import Path, PurePath

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from pkg_resources import resource_filename

import pdfstream.calibration as calib
import pdfstream.integration as integ
import pdfstream.io as io
import pdfstream.modeling as model
import pdfstream.visualization as vis


def integrate(
    poni_file: str, *img_files: str, bg_img_file: str = None,
    output_dir: str = ".", bg_scale: float = None, mask_setting: tp.Union[dict, str] = None,
    integ_setting: dict = None, plot_setting: tp.Union[dict, str] = None,
    img_setting: tp.Union[dict, str] = None
) -> tp.List[str]:
    """Conduct azimuthal integration on the two dimensional diffraction images.

    The image will be first subtracted by background if background image file is given. Then, it will be binned
    in azimuthal direction according to the geometry provided by the poni file. The pixels far away from the
    average in each bin will be masked. The mask will be applied on the background subtracted image and the
    image will be integrated again by the pyFAI. The polarization correction and pixel-splitting algorithm will
    be applied according to user settings before the integration. The results are saved as chi files.

    Parameters
    ----------
    poni_file : str
        The path to the poni file. It will be read by pyFAI.

    img_files : str
        The arbitrary number of paths to the image file. It will be read by fabio.

    bg_img_file : str
        The path to the background image file. It should have the same dimension as the data image. If None,
        no background subtraction will be done.

    output_dir : str
        The directory to save the chi data file. Default current working directory.

    bg_scale : float
        The scale of the background. Default 1

    mask_setting : dict or 'OFF'
        The settings for the auto-masking. See the arguments for mask_img (
        https://xpdacq.github.io/xpdtools/xpdtools.html?highlight=mask_img#xpdtools.tools.mask_img). To turn
        off the auto masking, enter "OFF".

    integ_setting : dict
        The settings for the integration. See the arguments for integrate1d (
        https://pyfai.readthedocs.io/en/latest/api/pyFAI.html#module-pyFAI.azimuthalIntegrator).

    plot_setting : dict or 'OFF'
        The keywords for the matplotlib.pyplot.plot (
        https://matplotlib.org/3.2.1/api/_as_gen/matplotlib.pyplot.plot.html). To turn off the plotting,
        enter "OFF".

    img_setting : dict or 'OFF'
        The keywords for the matplotlib.pyplot.imshow (
        https://matplotlib.org/3.2.1/api/_as_gen/matplotlib.pyplot.imshow.html). Besides, there is a key
        'z_score', which determines the range of the colormap. The range is mean +/- z_score * std in the
        statistics of the image. To turn of the image, enter "OFF".

    Returns
    -------
    chi_files : a list of strings
        The path to the output chi file.
    """
    if integ_setting is None:
        integ_setting = dict()
    ai = io.load_ai_from_poni_file(poni_file)
    bg_img = io.load_img(bg_img_file) if bg_img_file else None
    chi_paths = []
    for img_file in img_files:
        img = io.load_img(img_file)
        chi_name = Path(img_file).with_suffix('.chi').name
        chi_path = Path(output_dir).joinpath(chi_name)
        integ_setting.update({'filename': str(chi_path)})
        integ.get_chi(ai, img, bg_img, bg_scale=bg_scale, mask_setting=mask_setting,
                      integ_setting=integ_setting, plot_setting=plot_setting, img_setting=img_setting)
        chi_paths.append(str(chi_path))
    return chi_paths


def average(out_file: str, *img_files, weights: tp.List[float] = None) -> None:
    """Average the single channel image files with weights.

    Parameters
    ----------
    out_file : str
        The output file path. It will be the type as the first image in img_files.

    img_files : a tuple of str
        The image files to be averaged.

    weights : an iterable of floats
        The weights for the images. If None, images will not be weighted when averaged.
    """
    img_files: tp.Tuple[str]
    imgs = (io.load_img(_) for _ in img_files)
    avg_img = integ.avg_imgs(imgs, weights=weights)
    io.write_img(out_file, avg_img, img_files[0])
    return


def waterfall(
    *data_files: str, ax: Axes = None, mode: str = "line", normal: bool = True,
    stack: bool = True, gap: float = 0, texts: tp.List[str] = None, text_xy: tuple = None,
    label: str = None, minor_tick: tp.Union[int, None] = 2, legends: tp.List[str] = None,
    colors: tp.Iterable = None, show_fig: bool = True, **kwargs
) -> Axes:
    """Visualize the data in multiple data files in a waterfall or comparison plot.

    The data must be multi-columns matrix in a txt file. A header can be included in the file and it won't be
    read. The plots will be stacked in a waterfall if stack = True or overlapping together if stack = False. The
    visualization has different modes for different kinds of plotting. Currently, it supports

    'line' mode
        Each dataset is a single curve of x and y data.

    'fit' mode
        Each dataset contains three curves, one curve for data, one curve for the fit and a difference curve
        below it.

    Parameters
    ----------
    data_files : a list of file paths
        The file paths. Each file contains column data. The required format depends on the 'mode'.
        If mode = 'line', data = (x_array, y_array)
        If mode = 'fit', data = (x_array, y_array, ycalc_array)

    kwargs : optional
        The kwargs arguments for the plotting of each data. It depends on mode.
        If mode = 'line', kwargs in ('xy_kwargs',).
        If mode = 'fit', kwargs in ('xy_kwargs', 'xycalc_kwargs', 'xydiff_kwargs', 'xyzero_kwargs',
        'fill_kwargs', 'yzero').

    mode : str
        The plotting mode. Currently support 'line', 'fit'.

    ax : Axes
        The axes to visualize the data. If None, use current axes. (Not used for CLI)

    normal : bool
        If True, the second and the following rows in data will be normalized by (max - min). Else, do nothing.

    stack : bool
        If True, the second and the third rows will be shifted so that there will be a gap between data (
        waterfall plot). Else, the data will be plotted without shifting (comparison plot).

    gap : float
        The gap between the adjacent curves. It is defined by the nearest points in vertical direction.

    texts : an iterable of str
        The texts to annotate the curves. It has the same order as the curves.

    text_xy : tuple
        The tuple of x and y position of the annotation in data coordinates. If None, use the default in the
        'tools.auto_text'.

    label : str
        The label type used in automatic labeling. Acceptable types are listed in 'tools._LABELS'. If None,
        the label will be guessed according to the suffix of the fist file in the list.

    minor_tick : int
        How many parts that the minor ticks separate the space between the two adjacent major ticks. Default 2.
        If None, no minor ticks.

    legends : a list of str
        The legend labels for the curves.

    colors : an iterable of colors
        The color of the plots. If None, use default color cycle in rc.

    show_fig : bool
        If True, the figure will be pop out and shown. Else, stay in the cache.

    Returns
    -------
    ax : Axes
        The axes with the plot.
    """
    if len(data_files) == 0:
        raise ValueError("No input file.")
    dataset = (io.load_array(_) for _ in data_files)
    if label is None:
        label = PurePath(data_files[0]).suffix.replace('.', '')
    ax = vis.waterfall(
        dataset, ax=ax, mode=mode, normal=normal, stack=stack, gap=gap, texts=texts, text_xy=text_xy,
        label=label, minor_tick=minor_tick, legends=legends, colors=colors, **kwargs
    )
    if show_fig:
        plt.show(block=False)
    return ax


def visualize(
    data_file: str, ax: Axes = None, mode: str = "line", normal: bool = False,
    text: str = None, text_xy: tuple = None, label: str = None,
    minor_tick: int = 2, legends: tp.List[str] = None, color: tp.Iterable = None,
    show_fig: bool = True, **kwargs
) -> Axes:
    """Visualize the data in a single data file.

    The data must be multi-columns matrix in a txt file. A header can be included in the file and it won't be
    read. The visualization has different modes for different kinds of plotting. Currently, it supports

    'line' mode
        The single curve of x and y data.

    'fit' mode
        One curve for data, one curve for the fit and a difference curve below it.

    Parameters
    ----------
    data_file : file path
        The file path. The file contains column data. The required format depends on the 'mode'.
        If mode = 'line', data = (x_array, y_array)
        If mode = 'fit', data = (x_array, y_array, ycalc_array)

    kwargs : optional
        The kwargs arguments for the plotting of each data. It depends on mode.
        If mode = 'line', kwargs in ('xy_kwargs',).
        If mode = 'fit', kwargs in ('xy_kwargs', 'xycalc_kwargs', 'xydiff_kwargs', 'xyzero_kwargs',
        'fill_kwargs', 'yzero').

    mode : str
        The plotting mode.

    ax : Axes
        The axes to visualize the data. If None, use current axes. (Not used for CLI)

    normal : bool
        If True, the second and the following rows in data will be normalized by (max - min). Else, do nothing.
        Defulat False.

    text : str
        The text to annotate the curve.

    text_xy : tuple
        The tuple of x and y position of the annotation in data coordinates. If None, use the default in the
        'tools.auto_text'.

    label : str
        The label type used in automatic labeling. Acceptable types are listed in 'tools._LABELS'

    minor_tick : int
        How many parts that the minor ticks separate the space between the two adjacent major ticks. Default 2.
        If None, no minor ticks.

    legends : list
        The legend label for the curve.

    color : an iterable of colors
        The color of the plots. If None, use default color cycle in rc.

    show_fig : bool
        If True, the figure will be pop out and shown. Else, stay in the cache.

    Returns
    -------
    ax : Axes
        The axes with the plot.
    """
    data = io.load_array(data_file)
    if label is None:
        label = PurePath(data_file).suffix.replace('.', '')
    ax = vis.visualize(
        data, ax=ax, mode=mode, normal=normal, text=text, text_xy=text_xy, label=label,
        minor_tick=minor_tick, legends=legends, color=color, **kwargs)
    if show_fig:
        plt.show(block=False)
    return ax


def instrucalib(
    poni_file: str, img_file: str, cfg_file: str = None, stru_file: str = None, output_dir=".",
    fit_range: tp.Tuple[float, float, float] = (2.0, 60.0, 0.01),
    qdamp0: float = 0.04, qbroad0: float = 0.02,
    bg_img_file: str = None, bg_scale: float = None,
    mask_setting: tp.Union[dict, str] = None, integ_setting: dict = None,
    chi_plot_setting: tp.Union[dict, str] = None, img_setting: tp.Union[dict, str] = None,
    pdf_plot_setting: tp.Union[dict, str] = None, ncpu: int = None
):
    """Calibrate the 'qdamp' and 'qbroad' factor of the instrument in a pipeline process.

    A pipeline to do image background subtraction, auto masking, integration, PDF transformation and PDF
    modeling to calibrate the qdamp and qbroad. Also, the accuracy of the calibration is tested by the modeling.
    The output will be the processed data in 'iq', 'sq', 'fq', 'gr' files (depends on 'cfg' file), the fitting
    results in 'res' file, the refined structure in 'cif' file, the best fits data in 'fgr' file.

    Parameters
    ----------
    poni_file : str
        The path to the poni file. It will be read by pyFAI.

    img_file : str
        The path to the image file. It will be read by fabio.

    cfg_file : str
        The path to the PDF configuratino file. Usually, a 'cfg' file or a processed data file like 'gr' file
        with meta data in the header. If None, use the 'Ni_gr_file.gr' in 'pdfstream/test_data'.

    stru_file : str
        The path to the structure file. Usually, a 'cif' file. If None, use the 'Ni_cif_file.cif' in
        'pdfstream/test_data'.

    output_dir : str
        The directory to save all the outputs. Default current working directory.

    fit_range : tuple
        The rmin, rmax and rstep in the unit of angstrom.

    qdamp0 : float
        The initial value for the Q damping factor.

    qbroad0 : float
        The initial vluae for the Q broadening factor.

    bg_img_file : str
        The path to the background image file. It should have the same dimension as the data image. If None,
        no background subtraction will be done.

    bg_scale : float
        The scale for background subtraction. If None, use 1.

    mask_setting : dict or 'OFF'
        The settings for the auto-masking. See the arguments for mask_img (
        https://xpdacq.github.io/xpdtools/xpdtools.html?highlight=mask_img#xpdtools.tools.mask_img). To turn
        off the auto masking, enter "OFF".

    integ_setting : dict
        The settings for the integration. See the arguments for integrate1d (
        https://pyfai.readthedocs.io/en/latest/api/pyFAI.html#module-pyFAI.azimuthalIntegrator).

    img_setting : dict or 'OFF'
        The keywords for the matplotlib.pyplot.imshow (
        https://matplotlib.org/3.2.1/api/_as_gen/matplotlib.pyplot.imshow.html). Besides, there is a key
        'z_score', which determines the range of the colormap. The range is mean +/- z_score * std in the
        statistics of the image. To turn of the image, enter "OFF".

    chi_plot_setting : dict or 'OFF'
        The kwargs of chi data plotting. See matplotlib.pyplot.plot(
        https://matplotlib.org/3.2.1/api/_as_gen/matplotlib.pyplot.plot.html). If 'OFF', skip visualization.

    pdf_plot_setting : dict or 'OFF'
        The kwargs of pdf data plotting. See matplotlib.pyplot.plot(
        https://matplotlib.org/3.2.1/api/_as_gen/matplotlib.pyplot.plot.html). If 'OFF', skip visualization.

    ncpu : int
        The number of cpu used in parallel computing in the modeling. If None, no parallel computing.
    """
    if cfg_file is None:
        cfg_file = resource_filename('pdfstream', 'data/Ni_cfg_file.cfg')
    if stru_file is None:
        stru_file = resource_filename('pdfstream', 'data/Ni_cif_file.cif')
    ai = io.load_ai_from_poni_file(poni_file)
    img = io.load_img(img_file)
    pdfconfig = io.load_pdfconfig(cfg_file)
    stru = io.load_crystal(stru_file)
    bg_img = io.load_img(bg_img_file) if bg_img_file is not None else None
    pdfgetter, recipe = calib.calib_pipe(
        ai, img, pdfconfig, stru, fit_range=fit_range, qdamp0=qdamp0, qbroad0=qbroad0,
        bg_img=bg_img, bg_scale=bg_scale, mask_setting=mask_setting, integ_setting=integ_setting,
        chi_plot_setting=chi_plot_setting, img_setting=img_setting, pdf_plot_setting=pdf_plot_setting,
        ncpu=ncpu
    )
    img_path = PurePath(img_file)
    io.write_out(output_dir, img_path.name, pdfgetter)
    model.save(recipe, base_name=img_path.name, folder=output_dir)
    return
