import numpy as np
from scipy.linalg import sqrtm, block_diag
from scipy.stats import random_correlation

from pathlib import Path

from independent_vector_analysis.helpers_iva import _bss_isi
from independent_vector_analysis.consistent_iva import consistent_iva
from independent_vector_analysis.visualization import plot_scv_covs

from multiset_canonical_correlation_analysis import simulations, mcca


def scv_covs_with_noisy_blocks(N, K, indices, alpha):
    """
    Return SCV covariance matrices of dimension (K,K,N) that are generated as
    C = (1-alpha) v v.T + alpha Q Q^T,
    where Q is of dimensions (K,K), and v is a vector with 1s on the positions defined by indices


    Parameters
    ----------
    N : int
        number of SCVs

    K : int
        number of datasets

    R : int
        low rank of the model

    Returns
    -------
    scv_cov : np.ndarray
        Array of dimensions (K, K, N) that contains the SCV covariance matrices

    """

    vector = np.zeros((K, 1))
    vector[indices, 0] = 1
    L = np.random.randn(K, K)
    L /= np.linalg.norm(L, axis=1, keepdims=True)
    scv_cov = (1 - alpha) * (vector @ vector.T) + alpha * (L @ L.T)

    return scv_cov


def experiment_1(alpha=0.9):
    N = 5
    K = 20
    scv_cov = np.zeros((K, K, N))

    # SCVs have patterns
    cov = scv_covs_with_noisy_blocks(1, K, [0, 1, 2, 5, 7, 9, 10, 11, 13, 15], alpha=alpha)
    np.fill_diagonal(cov, 1)
    scv_cov[:, :, 0] = cov

    cov = scv_covs_with_noisy_blocks(1, K, [0, 1, 2, 3, 4, 5, 15, 16, 17], alpha=alpha)
    np.fill_diagonal(cov, 1)
    scv_cov[:, :, 1] = cov

    cov = scv_covs_with_noisy_blocks(1, K, [4, 5, 7, 8, 9, 17, 18, 19], alpha=alpha)
    np.fill_diagonal(cov, 1)
    scv_cov[:, :, 2] = cov

    cov = scv_covs_with_noisy_blocks(1, K, [9, 11, 13, 15, 16, 17, 19], alpha=alpha)
    np.fill_diagonal(cov, 1)
    scv_cov[:, :, 3] = cov

    cov = scv_covs_with_noisy_blocks(1, K, [0, 1, 4, 7, 11, 15], alpha=alpha)
    np.fill_diagonal(cov, 1)
    scv_cov[:, :, 4] = cov

    # plot_scv_covs(scv_cov)

    return scv_cov


def save_joint_isi(V, N, K, ortho, n_montecarlo, use_true_C_xx, algorithms, alpha=0.0):
    if use_true_C_xx:
        folder = 'true_C'
    else:
        folder = f'V_{V}'
    folder += f'_alpha_{alpha}_ortho_{ortho}'

    for run in range(n_montecarlo):
        print(f'Start run {run}...')

        scv_cov = experiment_1(alpha=alpha)

        filename = Path(Path(__file__).parent.parent, f'simulation_results/{folder}_true_run{run}.npy')
        np.save(filename, {'joint_isi': 0, 'scv_cov': scv_cov})

        X, A, S = simulations.generate_datasets_from_covariance_matrices(scv_cov, V, orthogonal_A=ortho)

        if use_true_C_xx:
            # true joint SCV covariance matrix
            joint_scv_cov = block_diag(*list(scv_cov.T))

            # make the permutation matrix
            P = np.zeros((N * K, N * K))
            for n in range(N):
                for k in range(K):
                    P[n + k * N, n * K + k] = 1

            # generate C_xx from true C_ss
            C_ss = P @ joint_scv_cov @ P.T
            A_joint = block_diag(*list(A.T)).T
            C_xx_all = A_joint @ C_ss @ A_joint.T
            C_xx = np.zeros((N, N, K, K), dtype=X.dtype)
            for k in range(K):
                for l in range(k, K):
                    C_xx[:, :, k, l] = C_xx_all[k * N:(k + 1) * N, l * N:(l + 1) * N]
                    C_xx[:, :, l, k] = C_xx[:, :, k, l].T  # R_xx is symmetric

        else:
            # calculate cross-covariance matrices of X
            C_xx = np.zeros((N, N, K, K), dtype=X.dtype)
            for k1 in range(K):
                for k2 in range(k1, K):
                    C_xx[:, :, k1, k2] = 1 / V * X[:, :, k1] @ X[:, :, k2].T
                    C_xx[:, :, k2, k1] = C_xx[:, :, k1, k2].T  # R_xx is Hermitian

            C_xx_all = np.zeros((N * K, N * K), dtype=X.dtype)
            for k in range(K):
                for l in range(k, K):
                    C_xx_all[k * N:(k + 1) * N, l * N:(l + 1) * N] = C_xx[:, :, k, l]
                    C_xx_all[l * N:(l + 1) * N, k * N:(k + 1) * N] = C_xx_all[k * N:(k + 1) * N,
                                                                     l * N:(l + 1) * N].T  # R_xx is symmetric

        # whiten datasets
        # whiten x^[k] -> y^[k] (using Mahalanobis whitening)
        A_whitened = np.zeros_like(A)
        X_whitened = np.zeros_like(X)
        for k in range(K):
            whitening = np.linalg.inv(sqrtm(C_xx[:, :, k, k]))
            A_whitened[:, :, k] = whitening @ A[:, :, k]
            X_whitened[:, :, k] = whitening @ X[:, :, k]

        # covariance of whitened data
        if use_true_C_xx:
            A_joint = block_diag(*list(A_whitened.T)).T
            C_xx_all = A_joint @ C_ss @ A_joint.T
            C_xx = np.zeros((N, N, K, K), dtype=X.dtype)
            for k in range(K):
                for l in range(k, K):
                    C_xx[:, :, k, l] = C_xx_all[k * N:(k + 1) * N, l * N:(l + 1) * N]
                    C_xx[:, :, l, k] = C_xx[:, :, k, l].T  # R_xx is symmetric
        else:
            # calculate cross-covariance matrices of X
            C_xx = np.zeros((N, N, K, K), dtype=X.dtype)
            for k1 in range(K):
                for k2 in range(k1, K):
                    C_xx[:, :, k1, k2] = 1 / V * X_whitened[:, :, k1] @ X_whitened[:, :, k2].T
                    C_xx[:, :, k2, k1] = C_xx[:, :, k1, k2].T  # R_xx is Hermitian

            C_xx_all = np.zeros((N * K, N * K), dtype=X.dtype)
            for k in range(K):
                for l in range(k, K):
                    C_xx_all[k * N:(k + 1) * N, l * N:(l + 1) * N] = C_xx[:, :, k, l]
                    C_xx_all[l * N:(l + 1) * N, k * N:(k + 1) * N] = C_xx_all[k * N:(k + 1) * N,
                                                                     l * N:(l + 1) * N].T  # R_xx is symmetric

        # randomly initialize a list of W_init
        W_init_list = []
        for i in range(20):
            W_init = np.random.randn(N, N, K)

            for k in range(K):
                W_init[:, :, k] = np.linalg.solve(sqrtm(W_init[:, :, k] @ W_init[:, :, k].T), W_init[:, :, k])
                # # add true inverse of A^-1 to the W_init
                # W_init[:, :, k] = 0.9 * np.linalg.inv(A_whitened[:, :, k]) + 0.1 * W_init[:, :, k]
                # W_init[:, :, k] = np.linalg.solve(sqrtm(W_init[:, :, k] @ W_init[:, :, k].T), W_init[:, :, k])

            W_init_list.append(W_init)

        # use W_init from most consistent IVA-G run as initialization for all algorithms
        ivag_results = consistent_iva(X_whitened, which_iva='iva_g', W_init=W_init_list, n_runs=10, A=A_whitened,
                                      R_xx=C_xx, whiten=False, parallel=False, opt_approach='gradient')
        W_init = ivag_results['W_init']

        # Newton IVA-G
        W_ivag_newton = consistent_iva(X_whitened, which_iva='iva_g', W_init=[W_init], n_runs=1, A=A_whitened,
                                       R_xx=C_xx, whiten=False, parallel=False)['W']

        # calculate SCV cov
        s_hat_cov = np.zeros((K, K, N))
        for n in range(N):
            # Efficient version of Sigma_n = 1/T * Y_n @ np.conj(Y_n.T) with Y_n = W_n @ X_n
            for k1 in range(K):
                for k2 in range(k1, K):
                    s_hat_cov[k1, k2, n] = W_ivag_newton[n, :, k1] @ C_xx[:, :, k1, k2] @ W_ivag_newton[n, :, k2]
                    s_hat_cov[k2, k1, n] = s_hat_cov[k1, k2, n]  # Sigma_n is symmetric

        filename = Path(Path(__file__).parent.parent,
                        f'simulation_results/{folder}_iva_g_newton_run{run}.npy')
        np.save(filename, {'joint_isi': _bss_isi(W_ivag_newton, A_whitened)[1], 'scv_cov': s_hat_cov})

        for algorithm_idx, algorithm in enumerate(algorithms):
            if algorithm is 'genvar':
                M, Epsilon = mcca.mcca(X_whitened, 'genvar', C_xx=C_xx_all, W_init=W_init)
                W = np.moveaxis(M, [0, 1, 2], [1, 0, 2])

            else:
                results = consistent_iva(X_whitened, which_iva=algorithm, W_init=[W_init], n_runs=1, A=A_whitened,
                                         R_xx=C_xx, whiten=False, parallel=False, opt_approach='gradient')
                W = results['W']

            # calculate SCV cov
            s_hat_cov = np.zeros((K, K, N))
            for n in range(N):
                # Efficient version of Sigma_n = 1/T * Y_n @ np.conj(Y_n.T) with Y_n = W_n @ X_n
                for k1 in range(K):
                    for k2 in range(k1, K):
                        s_hat_cov[k1, k2, n] = W[n, :, k1] @ C_xx[:, :, k1, k2] @ W[n, :, k2]
                        s_hat_cov[k2, k1, n] = s_hat_cov[k1, k2, n]  # Sigma_n is symmetric

            filename = Path(Path(__file__).parent.parent,
                            f'simulation_results/{folder}_{algorithm}_run{run}.npy')
            np.save(filename, {'joint_isi': _bss_isi(W, A_whitened)[1], 'scv_cov': s_hat_cov})
