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

from ..simulations import save_joint_isi, generate_scv_covs
from ..visualization import plot_results_for_different_samples, \
    plot_scv_covs_one_run, plot_results_for_different_alpha


def test_generate_scv_covs():
    generate_scv_covs(0.1)


def test_save_paper_results_alpha():
    np.random.seed(0)

    n_montecarlo = 50  # runs
    algorithms = ['iva_g', 'o-iva_g', 'd-o-iva_g', 'genvar']

    V = 10
    use_true_C_xx = True
    N = 5
    K = 20
    ortho = False
    for alpha in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
        save_joint_isi(V, N, K, ortho, n_montecarlo, use_true_C_xx, algorithms, alpha=alpha)


def test_save_paper_results_samples():
    np.random.seed(0)

    n_montecarlo = 50  # runs
    algorithms = ['iva_g', 'o-iva_g', 'd-o-iva_g', 'genvar']

    use_true_C_xx = False
    N = 5
    K = 20
    alpha = 0.7
    ortho = False
    for V in [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]:  # samples
        save_joint_isi(V, N, K, ortho, n_montecarlo, use_true_C_xx, algorithms, alpha=alpha)


def test_plot_results_for_different_alpha():
    n_montecarlo = 50

    N = 5
    K = 20
    ortho = False
    algorithms = ['iva_g', 'o-iva_g', 'd-o-iva_g', 'genvar', 'iva_g_newton']

    alpha_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    plot_results_for_different_alpha(alpha_values, N, K, ortho, n_montecarlo, algorithms, save=False)


def test_plot_results_for_different_samples():
    n_montecarlo = 50

    N = 5
    K = 20
    ortho = False
    algorithms = ['iva_g', 'o-iva_g', 'd-o-iva_g', 'genvar', 'iva_g_newton']

    V_values = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]
    plot_results_for_different_samples(V_values, N, K, ortho, n_montecarlo, algorithms, alpha=0.7, save=False)


def test_plot_results_for_different_subspace_structure():
    use_true_C_xx = True
    N = 5
    K = 20
    V = 100
    ortho = False
    algorithms = ['true', 'iva_g', 'o-iva_g', 'd-o-iva_g', 'genvar']

    run = 6
    plot_scv_covs_one_run(V, N, K, ortho, run, use_true_C_xx, algorithms, alpha=0.4, save=True)
