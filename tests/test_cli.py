from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
import numpy as np
import pytest

import pdfstream.cli as cli
import pdfstream.io as io


@pytest.mark.parametrize(
    'kwargs', [
        {'bg_img_file': None},
        {'mask_setting': {'alpha': 1.}},
        {'integ_setting': {'npt': 1024}},
        {'plot_setting': {'ls': '--'}},
        {'img_setting': {'vmin': 0}}
    ]
)
def test_integrate(db, kwargs):
    with TemporaryDirectory() as tempdir:
        img_file = Path(db['white_img_file'])
        chi_file = Path(tempdir).joinpath(img_file.with_suffix('.chi').name)
        _kwargs = {'bg_img_file': db['black_img_file'], 'output_dir': tempdir}
        _kwargs.update(kwargs)
        cli.integrate(db['Ni_poni_file'], str(img_file), **_kwargs)
        assert chi_file.exists()
    plt.close()


@pytest.mark.parametrize(
    'kwargs', [
        {},
        {'weights': [1, 1]}
    ]
)
def test_average(db, kwargs):
    with TemporaryDirectory() as tempdir:
        img_file = Path(tempdir).joinpath('average.tiff')
        cli.average(img_file, db['white_img_file'], db['white_img_file'], **kwargs)
        avg_img = io.load_img(img_file)
        white_img = io.load_img(db['white_img_file'])
        assert np.array_equal(avg_img, white_img)


@pytest.mark.parametrize(
    'keys,kwargs', [
        (['Ni_gr_file', 'Ni_gr_file'], {'mode': 'line', 'legends': ['Ni0', 'Ni1']}),
        (['Ni_gr_file', 'Ni_gr_file'], {'mode': 'line', 'stack': False}),
        (['Ni_gr_file', 'Ni_gr_file'], {'mode': 'line', 'xy_kwargs': {'color': 'black'}, 'texts': ['Ni0', 'Ni1']}),
        (['Ni_gr_file', 'Ni_gr_file'], {'mode': 'line', 'colors': ['r', 'b'], 'texts': ['Ni0', 'Ni1']}),
        (['Ni_fgr_file', 'Ni_fgr_file'], {'mode': 'fit', 'texts': ['Ni0', 'Ni1']}),
        (['Ni_fgr_file', 'Ni_fgr_file'], {'mode': 'fit', 'stack': False}),
        (['Ni_fgr_file', 'Ni_fgr_file'], {'mode': 'fit', 'xy_kwargs': {'color': 'black'}})
    ]
)
def test_waterfall(db, keys, kwargs):
    data_files = (db[key] for key in keys)
    cli.waterfall(*data_files, **kwargs)
    plt.close()


def test_waterfall_exception():
    with pytest.raises(ValueError):
        cli.waterfall(*tuple())


@pytest.mark.parametrize(
    'key,kwargs', [
        ('Ni_gr_file', {'mode': 'line', 'text': 'Ni', 'xy_kwargs': {'color': 'black'}}),
        ('Ni_gr_file', {'mode': 'line', 'legends': 'Ni', 'xy_kwargs': {'color': 'black'}}),
        ('Ni_fgr_file', {'mode': 'fit', 'text': 'Ni', 'xy_kwargs': {'color': 'black'}})
    ]
)
def test_visualize(db, key, kwargs):
    cli.visualize(db[key], **kwargs)
    plt.close()


def test_instrucalib(db):
    with TemporaryDirectory() as temp:
        cli.instrucalib(db['Ni_poni_file'], db['Ni_img_file'], output_dir=temp, fit_range=(2., 10., .1))
    plt.close()
