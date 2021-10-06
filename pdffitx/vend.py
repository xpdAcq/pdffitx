from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from pdfstream.vend.jittools import mask_ring_mean, mask_ring_median
from skbeam.core.accumulators.binned_statistic import BinnedStatistic1D
from skbeam.core.mask import margin

mask_ring_dict = {"median": mask_ring_median, "mean": mask_ring_mean}


def map_to_binner(pixel_map, bins, mask=None):
    """Transforms pixel map and bins into a binner

    Parameters
    ----------
    pixel_map: np.ndarray
        The map between pixels and values
    bins: np.ndarray
        The bins to use in the binner
    mask: np.ndarray, optional
        The mask for the pixel map

    Returns
    -------
    BinnedStatistic1D:
        The binner

    """
    if mask is not None:
        mask = mask.flatten()
    return BinnedStatistic1D(pixel_map.flatten(), bins=bins, mask=mask)


def generate_binner(geo, img_shape, mask=None):
    """Create a pixel resolution BinnedStats1D instance

    Parameters
    ----------
    geo : pyFAI.geometry.Geometry instance
        The calibrated geometry
    img_shape : tuple, optional
        The shape of the image, if None pull from the mask. Defaults to None.
    mask : np.ndarray, optional
        The mask to be applied, if None no mask is applied. Defaults to None.
    Returns
    -------
    BinnedStatistic1D :
        The configured instance of the binner.
    """
    return map_to_binner(*generate_map_bin(geo, img_shape), mask=mask)


def mask_img(
        img,
        binner,
        edge=30,
        lower_thresh=0.0,
        upper_thresh=None,
        alpha=2,
        auto_type="median",
        tmsk=None,
        pool=None,
):
    """
    Mask an image based off of various methods

    Parameters
    ----------
    img: np.ndarray
        The image to be masked
    binner : BinnedStatistic1D
        The binned statistics information
    edge: int, optional
        The number of edge pixels to mask. Defaults to 30. If None, no edge
        mask is applied
    lower_thresh: float, optional
        Pixels with values less than or equal to this threshold will be masked.
        Defaults to 0.0. If None, no lower threshold mask is applied
    upper_thresh: float, optional
        Pixels with values greater than or equal to this threshold will be
        masked.
        Defaults to None. If None, no upper threshold mask is applied.
    alpha: float, optional
        Then number of acceptable standard deviations, if tuple then we use
        a linear distribution of alphas from alpha[0] to alpha[1], if array
        then we just use that as the distribution of alphas. Defaults to 3.
        If None, no outlier masking applied.
    auto_type: {'median', 'mean'}, optional
        The type of binned outlier masking to be done, 'median' is faster,
        where 'mean' is more accurate, defaults to 'median'.
    tmsk: np.ndarray, optional
        The starting mask to be compounded on. Defaults to None. If None mask
        generated from scratch.
    pool : Executor instance
        A pool against which jobs can be submitted for parallel processing

    Returns
    -------
    tmsk: np.ndarray
        The mask as a boolean array. True pixels are good pixels, False pixels
        are masked out.

    """

    if tmsk is None:
        working_mask = np.ones(np.shape(img)).astype(bool)
    else:
        working_mask = tmsk.copy()
    if edge:
        working_mask *= margin(np.shape(img), edge)
    if lower_thresh is not None:
        working_mask *= (img >= lower_thresh).astype(bool)
    if upper_thresh is not None:
        working_mask *= (img <= upper_thresh).astype(bool)
    if alpha:
        working_mask *= binned_outlier(
            img,
            binner,
            alpha=alpha,
            tmsk=working_mask,
            mask_method=auto_type,
            pool=pool,
        )
    working_mask = working_mask.astype(np.int)
    return working_mask


def binned_outlier(
        img, binner, tmsk, alpha=3, mask_method="median", pool=None
):
    """Sigma Clipping based masking.

    Parameters
    ----------
    img : np.ndarray
        The image
    binner : BinnedStatistic1D instance
        The binned statistics information
    alpha : float, optional
        The number of standard deviations to clip, defaults to 3
    tmsk : np.ndarray, optional
        Prior mask. If None don't use a prior mask, defaults to None.
    mask_method : {'median', 'mean'}, optional
        The method to use for creating the mask, median is faster, mean is more
        accurate. Defaults to median.
    pool : Executor instance
        A pool against which jobs can be submitted for parallel processing

    Returns
    -------
    np.ndarray:
        The mask
    """
    if pool is None:
        pool = ThreadPoolExecutor(max_workers=24)
    # skbeam 0.0.12 doesn't have argsort_index cached
    idx = binner.argsort_index
    tmsk = tmsk.flatten()
    tmsk2 = tmsk[idx]
    vfs = img.flatten()[idx]
    pfs = np.arange(np.size(img))[idx]
    t = []
    i = 0
    for k in binner.flatcount:
        m = tmsk2[i: i + k]
        vm = vfs[i: i + k][m]
        if k > 0 and len(vm) > 0:
            t.append((vm, (pfs[i: i + k][m]), alpha))
        i += k
    p_err = np.seterr(all="ignore")
    # only run tqdm on mean since it is slow
    with pool as p:
        futures = [
            p.submit(mask_ring_dict[mask_method], *x)
            for x in t
        ]
    removals = []
    for f in as_completed(futures):
        removals.extend(f.result())
    np.seterr(**p_err)
    tmsk[removals] = False
    tmsk = tmsk.reshape(np.shape(img))
    return tmsk.astype(bool)


def generate_map_bin(geo, img_shape):
    """Create a q map and the pixel resolution bins

    Parameters
    ----------
    geo : pyFAI.geometry.Geometry instance
        The calibrated geometry
    img_shape : tuple, optional
        The shape of the image, if None pull from the mask. Defaults to None.

    Returns
    -------
    q : ndarray
        The q map
    qbin : ndarray
        The pixel resolution bins
    """
    r = geo.rArray(img_shape)
    q = geo.qArray(img_shape) / 10  # type: np.ndarray
    q_dq = geo.deltaQ(img_shape) / 10  # type: np.ndarray

    pixel_size = [getattr(geo, a) for a in ["pixel1", "pixel2"]]
    rres = np.hypot(*pixel_size)
    rbins = np.arange(np.min(r) - rres / 2., np.max(r) + rres / 2., rres / 2.)
    rbinned = BinnedStatistic1D(r.ravel(), statistic=np.max, bins=rbins)

    qbin_sizes = rbinned(q_dq.ravel())
    qbin_sizes = np.nan_to_num(qbin_sizes)
    qbin = np.cumsum(qbin_sizes)
    qbin[0] = np.min(q_dq)
    if np.max(q) > qbin[-1]:
        qbin[-1] = np.max(q)
    return q, qbin
