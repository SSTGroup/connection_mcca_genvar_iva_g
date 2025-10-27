# Simulations for our ICASSP2026 Paper

This package contains the code for reproducing the simulations of our paper:

Isabell Lehmann, B. Gabrielson, and T. Adali, **Revisiting the Connection between mCCA-genvar and IVA-G: Role of Orthogonality and Deflation**, submitted for *ICASSP 2026*.

## Installing this Package

The only pre-requisite is to have **Python 3** (version >=3.11) installed. This package can be
installed (optionally in a virtual environment) with:

    git clone https://github.com/SSTGroup/connection_mcca_genvar_iva_g
    cd connection_mcca_genvar_iva_g
    pip install -e .

Required third party packages will automatically be installed.


## Generating Simulations and Results

IMPORTANT NOTE:
A folder called _simulation_results_ must manually be created in _connection_mcca_genvar_iva_g_ before starting the simulations.

The simulations can be run by calling the functions `test_save_paper_results_alpha()` and `test_save_paper_results_samples()` in `test_experiments.py`.
After running the code, the folder *connection_mcca_genvar_iva_g/simulation_results* will contain the generated .npy files.

## Visualizing Results

The generated SCV covariance matrices can be visualized by calling the function `test_plot_scv_covs_one_run()` in `test_experiments.py`.

The results can be plotted by calling the functions `test_plot_results_for_different_alpha()` and `test_plot_results_for_different_samples()` in `test_experiments.py`.

## Contact

In case of questions, suggestions, problems etc. please send an email to isabell.lehmann@sst.upb.de, or open an issue here on Github.

## Citing

If you use this code in an academic paper, please cite [1]

    @inproceedings{Lehmann2026,
      title   = {Revisiting the Connection between mCCA-genvar and IVA-G: Role of Orthogonality and Deflation},
      author  = {Lehmann, Isabell and Gabrielson, Ben and Adali, T{\"u}lay},
      booktitle={submitted for review},
      year={2026}
      } 

[1] Isabell Lehmann, B. Gabrielson, and T. Adali, **Revisiting the Connection between mCCA-genvar and IVA-G: Role of Orthogonality and Deflation**, *submitted for review*, 2026.



