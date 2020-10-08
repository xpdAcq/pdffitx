import matplotlib.pyplot as plt

from pdfstream.calibration.main import calib_pipe


def test_calib_pipe(db):
    calib_pipe(
        db['ai'], db['Ni_img'], db['Ni_config'], db['Ni_stru'],
        fit_range=(2., 10., .1), qdamp0=0.04, qbroad0=0.02
    )
    plt.close()
