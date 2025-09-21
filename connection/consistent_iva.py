#     Copyright (c) <2025> <University of Paderborn>
#     Signal and System Theory Group, Univ. of Paderborn, https://sst-group.org/
#     https://github.com/SSTGroup/independent_vector_analysis
#
#     Permission is hereby granted, free of charge, to any person
#     obtaining a copy of this software and associated documentation
#     files (the "Software"), to deal in the Software without restriction,
#     including without limitation the rights to use, copy, modify and
#     merge the Software, subject to the following conditions:
#
#     1.) The Software is used for non-commercial research and
#        education purposes.
#
#     2.) The above copyright notice and this permission notice shall be
#        included in all copies or substantial portions of the Software.
#
#     3.) Publication, Distribution, Sublicensing, and/or Selling of
#        copies or parts of the Software requires special agreements
#        with the University of Paderborn and is in general not permitted.
#
#     4.) Modifications or contributions to the software must be
#        published under this license. The University of Paderborn
#        is granted the non-exclusive right to publish modifications
#        or contributions in future versions of the Software free of charge.
#
#     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
#     EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
#     OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
#     NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
#     HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
#     WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
#     FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
#     OTHER DEALINGS IN THE SOFTWARE.
#
#     Persons using the Software are encouraged to notify the
#     Signal and System Theory Group at the University of Paderborn
#     about bugs. Please reference the Software in your publications
#     if it was used for them.


import numpy as np

from independent_vector_analysis.helpers_iva import _bss_isi
from independent_vector_analysis.iva_g import iva_g
from independent_vector_analysis.consistent_iva import _run_selection_cross_isi

from .orthogonal_iva_g import orthogonal_iva_g
from .deflationary_orthogonal_iva_g import deflationary_iva_g


def consistent_iva(X, which_iva='iva_g', n_runs=20, W_init=None, **kwargs):
    """
    IVA is performed n_runs times with different initalizations.
    The most consistent demixing matrix along with the change in W for each iteration is returned,
    with the corresponding sources, mixing matrix, SCV covariance matrices.
    Consistence is measured by cross joint ISI, which can be returned optionally.


    Parameters
    ----------
    X : np.ndarray
        data matrix of dimensions N x T x K (sources x samples x datasets)

    which_iva : str, optional
        'iva_g', 'o-iva_g', or 'd-o-iva_g'

    n_runs : int, optional
        how many times iva is performed

    W_init : list
        list of initilization matrices, where each matrix is of dimensions N x N x K

    kwargs : list
        keyword arguments for the iva function


    Returns
    -------

    iva_results : dict
        - 'W' : estimated demixing matrix of most consistent run of dimensions N x N x K
        - 'W_change' : change in W for each iteration of most consistent run
        - 'S' : estimated sources of dimensions N x T x K
        - 'A' : estimated mixing matrix of dimensions N x N x K
        - 'scv_cov' : covariance matrices of the SCVs, of dimensions K x K x N
        - 'cross_isi' : cross joint joint_isi for each run
        - 'cost' : cost of each run
        - 'joint_isi' : joint_isi for each run if true A is supplied, else None
        - 'selected_run': index of the run with the lowest cross joint isi


    Notes
    -----
    Code written by Isabell Lehmann (isabell.lehmann at sst.upb.de)

    Reference:
    Long, Q., C. Jia, Z. Boukouvalas, B. Gabrielson, D. Emge, and T. Adali.
    "Consistent run selection for independent component analysis: Application
    to fMRI analysis." IEEE International Conference on Acoustics, Speech and
    Signal Processing (ICASSP), 2018.
    """

    W = []
    cost = []
    joint_isi = []
    W_change = []
    scv_cov = []
    for run in range(n_runs):
        if which_iva == 'iva_g':
            temp = orthogonal_iva_g(X, return_W_change=True, W_init=W_init[run], orthogonal=False, **kwargs)
        elif which_iva == 'o-iva_g':
            temp = orthogonal_iva_g(X, return_W_change=True, W_init=W_init[run], orthogonal='geodesic', **kwargs)
        elif which_iva == 'simple-o-iva_g':
            temp = orthogonal_iva_g(X, return_W_change=True, W_init=W_init[run], orthogonal='simple', **kwargs)
        elif which_iva == 'd-o-iva_g':
            temp = deflationary_iva_g(X, return_W_change=True, W_init=W_init[run], **kwargs)
        else:
            raise AssertionError("which_iva must be 'iva_g', 'o-iva_g, or 'd-o-iva_g'")
        W.append(temp[0])
        cost.append(temp[1])
        joint_isi.append(temp[3])
        W_change.append(temp[4])
        scv_cov.append(temp[2])

    cost = [c[-1] for c in cost]
    if joint_isi[0] is not None:
        joint_isi = [i[-1] for i in joint_isi]

    # use cross joint joint_isi to find most consistent run
    selected_run, _, cross_jnt_isi, _ = _run_selection_cross_isi(W)

    W = W[selected_run]
    W_change = W_change[selected_run]
    joint_isi = joint_isi[selected_run]
    cost = cost[selected_run]
    scv_cov = scv_cov[selected_run]
    W_init = W_init[selected_run]

    # get dimensions
    N, T, K = X.shape

    S = np.zeros((N, T, K))
    for k in range(K):
        S[:, :, k] = W[:, :, k] @ X[:, :, k]

    A_hat = np.zeros((N, N, K))
    for k in range(K):
        A_hat[:, :, k] = np.linalg.inv(W[:, :, k])

    # results
    iva_results = {'W': W, 'S': S, 'A': A_hat, 'scv_cov': scv_cov, 'W_change': W_change, 'W_init': W_init,
                   'cross_isi': cross_jnt_isi, 'cost': cost, 'joint_isi': joint_isi, 'selected_run': selected_run}

    return iva_results
