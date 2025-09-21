import numpy as np
import matplotlib.pyplot as plt
import matplot2tikz
from pathlib import Path
import pandas as pd


# grid for performance

def plot_results_for_different_NK(V, NK_pairs, R, alpha, ortho, use_true_C_xx, n_montecarlo, algorithms, save=False):
    # only varying parameters in this plot are N and K

    results = np.zeros((len(NK_pairs), len(algorithms), n_montecarlo))
    for idx, (N, K) in enumerate(NK_pairs):  # SCVs and datasets
        if use_true_C_xx:
            folder = 'true_C'
        else:
            folder = f'V_{V}'
        if R == 'K':
            folder += f'_N_{N}_K_{K}_R_{K}_alpha_{alpha}_ortho_{ortho}'
        else:
            folder += f'_N_{N}_K_{K}_R_{R}_alpha_{alpha}_ortho_{ortho}'
        for alg_idx, algorithm in enumerate(algorithms):
            for run in range(n_montecarlo):
                filename = Path(Path(__file__).parent.parent,
                                f'simulation_results/{folder}_{algorithm}_run{run}.npy')
                results[idx, alg_idx, run] = np.load(filename, allow_pickle=True).item()['joint_isi']

    plt.figure(figsize=(5, 2.5))

    for alg_idx, algorithm in enumerate(algorithms):
        plt.errorbar(np.arange(len(NK_pairs)), np.mean(results[:, alg_idx, :], axis=1),
                     np.std(results[:, alg_idx, :], axis=1),
                     linestyle=':', fmt='D', markersize=3, capsize=2, lw=1.1, label=f'{algorithm}')
    plt.xticks(np.arange(len(NK_pairs)), NK_pairs, fontsize=12)
    plt.xlabel(r'rank $(N,K)$', fontsize=12)
    plt.ylim([-0.006, .606])
    plt.yticks(np.arange(0, 0.61, 0.2), fontsize=12)
    plt.ylabel('jISI', fontsize=12)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    if save:
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.2)
        matplot2tikz.save(f'joint_isi_NK.tex', encoding='utf8', axis_width='7.5cm',
                          axis_height='5cm', standalone=True)
        plt.savefig(f'joint_isi_NK.pdf')
    else:
        plt.title(f'joint_isi for different values of (N,K)')
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.2)
        plt.show()


def plot_results_for_different_samples(V_values, N, K, ortho, n_montecarlo, algorithms, alpha=0.9, rho_d=0.1,
                                       save=False):
    results = np.zeros((len(V_values), len(algorithms), n_montecarlo))
    for V_idx, V in enumerate(V_values):  # SCVs and datasets
        folder = f'V_{V}_alpha_{alpha}_ortho_{ortho}'
        for alg_idx, algorithm in enumerate(algorithms):
            for run in range(n_montecarlo):
                filename = Path(Path(__file__).parent.parent,
                                f'simulation_results/{folder}_{algorithm}_run{run}.npy')
                results[V_idx, alg_idx, run] = np.load(filename, allow_pickle=True).item()['joint_isi']

    plt.figure(figsize=(5, 2.5))

    # label
    V_labels = [100, 1000, 10000, 100000]
    V_label_values = [f'${V_value}$' for V_value in V_labels]

    alg_names = {'n-o-iva_g': 'IVA-G', 'o-iva_g': 'o-IVA-G', 'd-o-iva_g': 'd-o-IVA-G', 'genvar': 'genvar',
                 'iva_g_newton': 'IVA-G (Newton)'}

    plt.axhline(y=0.05, color='tab:gray', linestyle=':', linewidth=1.1)
    for alg_idx, algorithm in enumerate(algorithms):
        plt.errorbar(V_values, np.mean(results[:, alg_idx, :], axis=1),
                     np.std(results[:, alg_idx, :], axis=1),
                     linestyle=':', fmt='D', markersize=4, capsize=1.8, lw=1.2, label=f'{alg_names[algorithm]}')
    plt.legend()  # loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xscale('log')
    plt.xticks(V_labels, V_label_values, fontsize=12)
    plt.xlabel(r'number of samples $V$', fontsize=12)
    plt.ylim([-0.025, 0.525])
    plt.yticks([0, 0.25, 0.5], ['$0$', '$0.25$', '$0.5$'], fontsize=12)
    plt.ylabel('jISI', fontsize=12)

    if save:
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.2)
        matplot2tikz.save(f'joint_isi_samples.tex', encoding='utf8', axis_width='9cm',
                          axis_height='5cm', standalone=True)
        plt.savefig(f'joint_isi_samples.pdf')
    else:
        plt.title(f'joint_isi for different number of samples')
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.2)
        plt.show()


def plot_results_for_different_alpha(alpha_values, N, K, ortho, n_montecarlo, algorithms, save=False):
    results = np.zeros((len(alpha_values), len(algorithms), n_montecarlo))
    for alpha_idx, alpha in enumerate(alpha_values):  # SCVs and datasets
        folder = f'true_C_alpha_{alpha}_ortho_{ortho}'
        for alg_idx, algorithm in enumerate(algorithms):
            for run in range(n_montecarlo):
                filename = Path(Path(__file__).parent.parent,
                                f'simulation_results/{folder}_{algorithm}_run{run}.npy')
                results[alpha_idx, alg_idx, run] = np.load(filename, allow_pickle=True).item()['joint_isi']

    plt.figure(figsize=(5, 2.5))

    # label
    alpha_labels = [0.1, 0.3, 0.5, 0.7, 0.9]
    alpha_label_values = [f'${alpha_value}$' for alpha_value in alpha_labels]

    alg_names = {'n-o-iva_g': 'IVA-G', 'iva_g_newton': 'IVA-G (Newton)', 'o-iva_g': 'o-IVA-G', 'd-o-iva_g': 'd-o-IVA-G',
                 'genvar': 'genvar'}

    plt.axhline(y=0.05, color='tab:gray', linestyle=':', linewidth=1.1)
    for alg_idx, algorithm in enumerate(algorithms):
        plt.errorbar(alpha_values, np.mean(results[:, alg_idx, :], axis=1),
                     np.std(results[:, alg_idx, :], axis=1),
                     linestyle=':', fmt='D', markersize=4, capsize=1.8, lw=1.2, label=f'{alg_names[algorithm]}')
    plt.legend()  # loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xticks(alpha_labels, alpha_label_values, fontsize=12)
    plt.xlabel(r'$\alpha$', fontsize=12)
    plt.ylim([-0.025, 0.525])
    plt.yticks([0, 0.25, 0.5], ['$0$', '$0.25$', '$0.5$'], fontsize=12)
    plt.ylabel('jISI', fontsize=12)

    if save:
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.2)
        matplot2tikz.save(f'joint_isi_alpha.tex', encoding='utf8', axis_width='9cm',
                          axis_height='5cm', standalone=True)
        plt.savefig(f'joint_isi_alpha.pdf')
    else:
        plt.title(f'joint_isi for different values of alpha')
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.2)
        plt.show()


def plot_one_run_scv_covs_checkerboard(V, N, K, ortho, run, use_true_C_xx, algorithms, alpha=0.9, save=False):
    if use_true_C_xx:
        folder = 'true_C'
    else:
        folder = f'V_{V}'
    folder += f'_alpha_{alpha}_ortho_{ortho}'

    n_cols = N
    n_rows = len(algorithms)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.2 * n_cols, 1.2 * n_rows), layout='constrained')

    alg_names = {'iva_g_newton': 'IVA-G (Newton)', 'o-iva_g': 'o-IVA-G', 'd-o-iva_g': 'd-o-IVA-G', 'genvar': 'genvar',
                 'n-o-iva_g': 'IVA-G', 'true': 'true'}

    # Add data to image grid and plot
    for algorithm_idx, algorithm in enumerate(algorithms):
        filename = Path(Path(__file__).parent.parent,
                        f'simulation_results/{folder}_{algorithm}_run{run}.npy')
        res = np.load(filename, allow_pickle=True).item()
        scv_cov = res['scv_cov']
        # plot_eigenvalues(scv_cov,title='true covariance matrices', filename=Path(Path(__file__).parent.parent.parent,
        #              f'simulation_results/{folder}/eigvals.png'))

        for scv_idx in range(scv_cov.shape[2]):
            im = axes[algorithm_idx, scv_idx].imshow(np.abs(scv_cov[:, :, scv_idx]), vmin=0, vmax=1, cmap='hot')
            axes[algorithm_idx, scv_idx].set_xticks([])
            axes[algorithm_idx, scv_idx].set_yticks([])

            # Minor ticks for grid
            axes[algorithm_idx, scv_idx].set_xticks(np.arange(-.5, scv_cov.shape[0], 1), minor=True)
            axes[algorithm_idx, scv_idx].set_yticks(np.arange(-.5, scv_cov.shape[0], 1), minor=True)
            # Gridlines based on minor ticks
            axes[algorithm_idx, scv_idx].grid(which='minor', color='k', linestyle='-', linewidth=0.1)
            # Remove minor ticks
            axes[algorithm_idx, scv_idx].tick_params(which='minor', bottom=False, left=False)

        axes[algorithm_idx, 0].set_ylabel(
            f'{alg_names[algorithm]} \\\\ jISI={res['joint_isi']:.1e}',
            # f' \n $\sum$logdet = {np.sum(np.linalg.slogdet(scv_cov.T)[1]):2.1f}',
            rotation=90, fontsize=9, labelpad=10, va='center')  # ,fontweight='bold')
        # f'{algorithm}', rotation=0, labelpad=22, va='center', fontweight='bold')

    for scv_idx in range(N):
        axes[0, scv_idx].set_title(
            # f'SCV {scv_idx + 1}:  \n logdet = {np.linalg.slogdet(scv_cov[:, :, scv_idx])[1]:2.1f}', fontsize=9,
            '{\Large $\mathbf{C}_{\mathbf{s}_' + f'{scv_idx + 1}' + '}$}', fontsize=8,
            pad=4)
    fig.colorbar(im, ax=axes[:, -1], pad=0.15, fraction=0.5)

    if not save:
        if use_true_C_xx:
            plt.suptitle('using true covariance matrix')
        else:
            plt.suptitle(f'V={V} samples')
        plt.show()
    else:
        matplot2tikz.save(f'scv_covs_{folder}.tex', encoding='utf8', standalone=True, axis_width='3.7cm',
                          axis_height='3.7cm', override_externals=True,
                          extra_axis_parameters={'xtick=\empty', 'ytick=\empty',
                                                 'ylabel style={align=center}'})
        plt.savefig(f'SCVcovs_{folder}.png', dpi=1000)
