"""The functions used in the command line interface. The input and output are all files."""
import matplotlib.pyplot as plt
import pdfstream.io as pio
import pdfstream.transformation.io as tio
import typing as tp
from pathlib import PurePath
from pkg_resources import resource_filename

import pdffitx.calibration as calib
import pdffitx.io as io
import pdffitx.modeling as md


def instrucalib(
    poni_file: str, img_file: str, cfg_file: str = None, stru_file: str = None, output_dir=".",
    fit_range: tp.Tuple[float, float, float] = (2.0, 60.0, 0.01),
    qdamp0: float = 0.04, qbroad0: float = 0.02,
    bg_img_file: str = None, bg_scale: float = None,
    mask_setting: tp.Union[dict, str] = None, integ_setting: dict = None,
    chi_plot_setting: tp.Union[dict, str] = None, img_setting: tp.Union[dict, str] = None,
    pdf_plot_setting: tp.Union[dict, str] = None, ncpu: int = None,
    show: bool = False
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

    show : bool
        If True, show figures.
    """
    if cfg_file is None:
        cfg_file = resource_filename('pdffitx', 'data/Ni_cfg_file.cfg')
    if stru_file is None:
        stru_file = resource_filename('pdffitx', 'data/Ni_cif_file.cif')
    ai = pio.load_ai_from_poni_file(poni_file)
    img = pio.load_img(img_file)
    pdfconfig = tio.load_pdfconfig(cfg_file)
    stru = io.load_crystal(stru_file)
    bg_img = pio.load_img(bg_img_file) if bg_img_file is not None else None
    pdfgetter, recipe = calib.calib_pipe(
        ai, img, pdfconfig, stru, fit_range=fit_range, qdamp0=qdamp0, qbroad0=qbroad0,
        bg_img=bg_img, bg_scale=bg_scale, mask_setting=mask_setting, integ_setting=integ_setting,
        chi_plot_setting=chi_plot_setting, img_setting=img_setting, pdf_plot_setting=pdf_plot_setting,
        ncpu=ncpu
    )
    img_path = PurePath(img_file)
    tio.write_pdfgetter(output_dir, img_path.name, pdfgetter)
    md.save(recipe, base_name=img_path.name, folder=output_dir)
    if show:
        plt.show()
    return
