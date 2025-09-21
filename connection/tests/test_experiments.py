import numpy as np

from ..icassp_simulations import save_joint_isi, experiment_1
from ..icassp_visualization import plot_results_for_different_samples, \
    plot_one_run_scv_covs_checkerboard, plot_results_for_different_alpha


def test_experiment_1():
    experiment_1(0.1)


def test_save_paper_results_alpha_09():
    np.random.seed(0)

    n_montecarlo = 50  # runs
    algorithms = ['n-o-iva_g', 'iva_g', 'o-iva_g', 'd-o-iva_g', 'genvar']

    V = 10
    use_true_C_xx = True
    N = 5
    K = 20
    ortho=False
    alpha = 0.9
    save_joint_isi(V, N, K, ortho, n_montecarlo, use_true_C_xx, algorithms, alpha=alpha)

def test_save_paper_results_samples_100000():
    np.random.seed(0)

    n_montecarlo = 50  # runs
    algorithms = ['n-o-iva_g', 'iva_g', 'o-iva_g', 'd-o-iva_g', 'genvar']

    use_true_C_xx = False
    N = 5
    K = 20
    alpha = 0.7
    ortho = False
    # for V in [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]:  # samples
    V=100000
    save_joint_isi(V, N, K, ortho, n_montecarlo, use_true_C_xx, algorithms, alpha=alpha)


def test_plot_results_for_different_alpha():
    n_montecarlo = 50

    N = 5
    K = 20
    ortho = False
    algorithms = ['n-o-iva_g', 'o-iva_g', 'd-o-iva_g', 'genvar', 'iva_g_newton']

    alpha_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    plot_results_for_different_alpha(alpha_values, N, K, ortho, n_montecarlo, algorithms, save=False)


def test_plot_results_for_different_samples():
    n_montecarlo = 50

    N = 5
    K = 20
    ortho = False
    algorithms = ['n-o-iva_g', 'o-iva_g', 'd-o-iva_g', 'genvar', 'iva_g_newton']

    V_values = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]
    plot_results_for_different_samples(V_values, N, K, ortho, n_montecarlo, algorithms, alpha=0.7, save=False)


def test_plot_results_for_different_subspace_structure():
    use_true_C_xx = True
    N = 5
    K = 20
    V = 100
    ortho = False
    algorithms = ['true', 'n-o-iva_g', 'o-iva_g', 'd-o-iva_g', 'genvar']

    run = 6
    plot_one_run_scv_covs_checkerboard(V, N, K, ortho, run, use_true_C_xx, algorithms, alpha=0.4, save=True)
