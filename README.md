# FedCVAWA + GraphEffiNet

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c.svg)](https://pytorch.org/)

Official PyTorch implementation of **GraphEffiNet** and **FedCVAWA** from:

> A. Khandakar et al.,
> *Embedded Federated Learning for Privacy-Aware Road Condition Monitoring with Graph-Augmented Vision on Vehicular Hardware* (under review).

## Contents

```
FedCVAWA-GraphEffiNet/
├── grapheffinet/
│   ├── __init__.py
│   ├── layers.py        # patch graph, GCN, Swin blocks, spatial self-attention
│   └── model.py         # EfficientNet-B0 branch, GraphEffiNet, ModelConfig
├── fedcvawa/
│   ├── __init__.py
│   ├── client.py        # top-K compression, local update, gradient stability
│   └── aggregation.py   # Stage 1 verification, Stage 2 adaptive weighting
├── configs/default.yaml # paper hyperparameters (Table 1)
├── DATASET.md           # dataset description and download link
├── LICENSE
├── pyproject.toml
└── requirements.txt
```


## Installation

```bash
git clone https://github.com/<your-username>/FedCVAWA-GraphEffiNet.git
cd FedCVAWA-GraphEffiNet
pip install -e .
```

Install the PyTorch build matching your CUDA version from [pytorch.org](https://pytorch.org/get-started/locally/) first if you need GPU support.


See [DATASET.md](DATASET.md). A sample of the external validation dataset is available on [Google Drive](https://drive.google.com/file/d/1VaxI7T54LuosRZxvCoqnssUc3q3slA_e/view?usp=sharing). RoadSens 4M dataset Figshare Link: https://figshare.com/articles/dataset/RoadSens-4M_Dataset/30341143.

## Citation

### It will be updated soon (after paper published)

```bibtex
@article{khandakar2026fedcvawa,
  title   = {Embedded Federated Learning for Privacy-Aware Road Condition Monitoring
             with Graph-Augmented Vision on Vehicular Hardware},
  author  = {Khandakar, Amith and others},
  note    = {Under review},
  year    = {2026}
}
```

## Acknowledgment

Funded by the Qatar University Internal Grant QUCG-CENG-26/27-1050.

## License

[MIT License](LICENSE). Copyright (c) 2026 Dr. Amith Khandakar.

## Contact

Dr. Amith Khandakar, College of Engineering, Qatar University. amitk@qu.edu.qa
