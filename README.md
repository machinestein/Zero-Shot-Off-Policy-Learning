# [Zero-Shot Off-Policy Learning](https://arxiv.org/pdf/2602.01962)
by [Arip Asadulaev](https://arip.page) and [Maksim Bobrin](https://github.com/skylooop) et.al.
# Overview
This repository provides a PyTorch implementation of different baseline methods used in paper. It also provides a framework for training, evaluating, and comparing unsupervised zero-shot RL methods on proprioceptive- and pixel-based environments from the [DeepMind Control Suite](https://github.com/google-deepmind/dm_control) and [OGBench](https://github.com/seohongpark/ogbench). The implementation of baselines is discussed in detail in the [paper](https://arxiv.org/abs/2510.00739).

# Quick Start

## Installation

We use [uv](https://github.com/astral-sh/uv) to manage dependencies. After [installation](https://docs.astral.sh/uv/getting-started/installation/), simply `cd` into the `td_jepa` directory and run
```
uv sync --all-extras
```
This will create a virtual environment in `.venv` with the dependencies required by this project. You can activate it explicitly by `source .venv/bin/activate`.

## Downloading the data

We provide scripts for downloading and processing [ExORL](https://github.com/denisyarats/exorl) and [OGBench](https://github.com/seohongpark/ogbench) datasets:
```
# OGBench
uv run -m scripts.data_processing.ogbench.extract_all --output_folder your_path_here
# ExORL
uv run -m scripts.data_processing.exorl.download --output_folder exorl_path_here
```

## Reproducing the experiments
### Toy2D
For showcasing failure mode FB on producing bad representation of new tasks, you need to run training on donut support dataset with
```
uv run fb_training_donut.py
```
This will save the pretrained FB model to the `fb_donut_model` directory.
Then, to visualize the results, run
```
uv run fb_toy2d_fb_visualize.py
```
This will save the results to the `fb_toy2d_simplified` directory.

### ZOL implementation
ZOL implementation is directly implemented in the `metamotivo/agents/fb/agent.py` with corresponding methods, which implement latent policy search.

After pretraining FB, you can sweep over different ZOL hyperparameters to choose best ones in the
```
uv run scripts/sweep_zol_ogbench.py
```
Or 
```
uv run scripts/sweep_zol_ogbench.py
```

### LoLA and ReLA
For final scores of LoLA and ReLA and best ZOL parameters, together with baseline, you need to run 
```
uv run scripts/eval_zol_dmc_all.py
```
Or 
```
uv run scripts/eval_zol_ogbench_all.py
```

### Baselines
In order to train other BFM baselines (HILP/TD-JEPA), you need to run corresponding scripts in `scripts/train` and find required launch script of model. By default, we log into wandb.
The launch options can be found in the bottom of each script.

This will sequentially run through a grid of experiments. Adding a `first_only` flag will only run the first experiment in the grid. By default, jobs are executed locally.

### Notebooks
Additionally, for visualizing the results, we provide notebooks in `notebooks` directory, together with the code for reproducing the results from the paper.

The current codebase is based on the [TD_JEPA repository](https://github.com/facebookresearch/td_jepa).
