from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt

import pdffitx.cli as cli


def test_instrucalib(db):
    with TemporaryDirectory() as temp:
        cli.instrucalib(db['Ni_poni_file'], db['Ni_img_file'], output_dir=temp, fit_range=(2., 10., .1))
    plt.close()
