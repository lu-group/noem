> Under Construction

# Neural-Operator Element Method (NOEM)

The data and code for the paper [Neural-operator element method: Efficient and scalable finite element method enabled by reusable neural operators](https://arxiv.org/abs/2506.18427).


## Code

This repository contains reference implementations for the **Neural-Operator Element Method (NOEM)**—a hybrid approach that combines the Finite Element Method (FEM) with reusable neural operators for efficient numerical simulations of partial differential equations (PDEs).

### Setup

Install dependencies (Python + pip):

```bash
python -m pip install -r requirements.txt
```

### Directory layout

- `pedagogical_example/`: Pedagogical examples from the Methods section.
- `multiscale_1d_problem/`: 1D multiscale problems (Results section).
- `heat_transfer/`: Heat transfer example (Results section).
- `darcy_flow/`: Darcy flow example (Results section).
- `convexity_test/`, `uq_test/`: Additional test cases (Supplementary Information).

### Running experiments

Most experiments are driven by a `run.py` script. Many scripts rely on files referenced by **relative paths**, so it is recommended to `cd` into the corresponding folder first and then run `python run.py`.

Examples (from the repository root):

```bash
cd pedagogical_example/quadratic_coefficient
python run.py
```

Available entry points:

- `pedagogical_example/quadratic_coefficient/run.py`
- `pedagogical_example/random_coefficient_functions/run.py`
- `multiscale_1d_problem/multiscale_coefficient/run.py`
- `multiscale_1d_problem/multiscale_source_term/run.py`
- `multiscale_1d_problem/multiscale_coefficient/results_fig_d/run.py`
- `heat_transfer/run.py`
- `darcy_flow/run.py`
- `convexity_test/run.py`
- `uq_test/run.py`

## Cite this work

If you use this data or code for academic research, you are encouraged to cite the following paper:

```
@article{ouyang2025neural,
  author  = {Ouyang, Weihang and Shin, Yeonjong and Liu, Si-Wei and Lu, Lu},
  title   = {Neural-operator element method: Efficient and scalable finite element method enabled by reusable neural operators},
  journal = {Nature Computational Science (Accepted)},
  year    = {2026},
  doi     = {https://doi.org/10.48550/arXiv.2506.18427}
}
```

## Questions

To get help on how to use the data or code, simply open an issue in the GitHub "Issues" section.
