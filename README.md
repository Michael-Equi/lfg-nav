# Navigation with Large Language Models: Semantic Guesswork as a Heuristic for Planning (LFG)

#### Dhruv Shah*, Michael Equi*, Błażej Osiński, Fei Xia, Brian Ichter, Sergey Levine

_Berkeley AI Research_

[Project Page](https://sites.google.com/view/lfg-nav/home) | [arXiV](https://arxiv.org/abs/2310.10103) | [Summary Video](https://youtu.be/jUqYQOW7E-4?si=iQRkrKGpfvn-BUJQ)

<a href="https://colab.research.google.com/github/Michael-Equi/lfg-demo/blob/main/lfg_demo_colab.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" height=22.5></a> 

## Introduction

This repository contains code used in *Navigation with Large Language Models: Semantic Guesswork as a Heuristic for Planning* by Dhruv Shah, Michael Equi, Błażej Osiński, Fei Xia, Brian Ichter, and Sergey Levine.

* `jupyter_experiment.ipynb` - contains the prompts and code used to score the frontiers.
* `lfg_demo.ipynb` - notebook with examples of using the language scoring to score different frontiers.
* `lfg_demo_colab.ipynb` - colab version of the above notebook. You can easily [run it in your browser](https://colab.research.google.com/github/Michael-Equi/lfg-demo/blob/main/lfg_demo_colab.ipynb)!


## Installation

To run locally, install the package:

```pip install -r requirements.txt```

Then simply open `lfg_demo.ipynb` in jupyter notebook.

## LLMs APIs

To use the API locally please provide an API key and organization ID in a `.env` file in the root directory of the repository.
```
OPENAI_API_KEY =  sk-***
OPENAI_ORG =  org-***
```

## Citation

If you find this work useful, please consider citing:

```
@misc{shah2023lfg,
      title={Navigation with Large Language Models: Semantic Guesswork as a Heuristic for Planning}, 
      author={Dhruv Shah and Michael Equi and Blazej Osinski and Fei Xia and Brian Ichter and Sergey Levine},
      year={2023},
      eprint={2310.10103},
      archivePrefix={arXiv},
      primaryClass={cs.RO}
}
```
