> Under Construction

# Neural-Operator Element Method (NOEM)

The data and code for the paper [Neural-operator element method: Efficient and scalable finite element method enabled by reusable neural operators](https://arxiv.org/abs/2506.18427).


## Code

This repository contains the implementation of the **Neural-Operator Element Method (NOEM)**, a hybrid computational approach that combines the Finite Element Method (FEM) with neural operators for efficient numerical simulations of partial differential equations (PDEs).

The code is organized into several main directories:
- `ex1/`, `ex2/`, `ex3/`, `ex4/`: Example problems demonstrating NOEM applications whose setups can be found in the paper
- `data_driven_training/`: Training scripts for neural operator models
- `convexity_test/`, `uq_test/`: Additional test cases in the appendix

To run the code:
1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Navigate to the specific example directory and run the corresponding scripts.

## Cite this work

If you use this data or code for academic research, you are encouraged to cite the following paper:

```
@article{ouyang2025neural,
  title={Neural-operator element method: Efficient and scalable finite element method enabled by reusable neural operators},
  author={Ouyang, Weihang and Shin, Yeonjong and Liu, Si-Wei and Lu, Lu},
  journal={arXiv preprint arXiv:2506.18427},
  year={2025}
}
```

## Questions

To get help on how to use the data or code, simply open an issue in the GitHub "Issues" section.
