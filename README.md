# Neural-Operator Element Method (NOEM)

## Overview

This repository contains the implementation of the **Neural-Operator Element Method (NOEM)**, a novel hybrid computational approach that synergistically combines the Finite Element Method (FEM) with neural operators to achieve efficient and scalable numerical simulations of partial differential equations (PDEs).

NOEM addresses the computational challenges of traditional FEM by leveraging Neural-Operator Elements (NOEs) in subdomains where dense meshing would otherwise be required, significantly reducing computational costs while maintaining accuracy.

## Cite this work

If you use this data or code for academic research, you are encouraged to cite the following paper:

**"Neural-operator element method: Efficient and scalable finite element method enabled by reusable neural operators"**

*Authors: Weihang Ouyang, Yeonjong Shin, Si-Wei Liu, Lu Lu*

Paper: [arXiv:2506.18427](https://arxiv.org/html/2506.18427v1)

## Key Features

- **Hybrid Approach**: Combines the robustness of FEM with the efficiency of neural operators
- **Scalable**: Reduces computational complexity for multiscale and complex geometry problems
- **Reusable**: Pre-trained neural operators can be reused across different problem instances
- **Versatile**: Handles nonlinear PDEs, multiscale problems, complex geometries, and discontinuous coefficient fields

## Installation

### Prerequisites

- Python 3.8 or higher
- PyTorch 1.9.0 or higher

### Install required dependencies:
```bash
pip install -r requirements.txt
```

### Dependencies

- `torch>=1.9.0` - Deep learning framework
- `torchvision>=0.10.0` - Computer vision utilities
- `numpy>=1.21.0` - Numerical computations
- `matplotlib>=3.4.0` - Plotting and visualization
- `scipy>=1.7.0` - Scientific computing
- `pandas>=1.3.0` - Data manipulation
- `tqdm>=4.62.0` - Progress bars
- `scikit-learn>=1.0.0` - Machine learning utilities

## License

This project is licensed under the Apache License Version 2.0 - see the LICENSE file for details.
