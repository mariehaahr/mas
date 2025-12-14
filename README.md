# Multi-agent Systems 

## Project Overview

Large Language models (LLMs) are increasingly deployed in multi-agent systems (MAS) where agents share information and coordinate decisions. 
However, most existing work focuses on homogenous settings, leaving little understanding of how diverse agents behave when interacting.
This gap matters because heterogeneous agents differ in architecture, training data, and label biases, which may shape how strongly they influence one another, an effect homogenous MAS evaluation cannot reveal.
In this paper we introduce a framework for quantifying directional influence between LLM agents by comparing an agent's individual behaviour on a task, to it's behaviour after interaction with another agent on the same task.
The change in behaviour is measured by the positive prediction rate before and after interaction.
Across six different models, we find that influence is widespread and often asymmetric. Some agents consistently shift others while remaining resistant themselves.
When controlling for label biases, new patterns emerge that show how bias alignment amplifies influence, even for highly confident agents. 
These findings show that heterogeneous MAS can exhibit complex and sometimes unexpected influence dynamics, which do not reduce to simple weak/strong model hierarchies.
Because such behaviours remain hidden in isolated evaluations, accounting for interaction effects is essential when aiming to build reliable and value-aligned collaborative MAS.


## Set up environment

This project is configured to run on NVIDIA GPUs and has been developed and tested using CUDA 12.6. 
If you plan to run this on a different CUDA version then you might need to adjust the environment. 

Depending on your preferred method the environment can be set up using either uv or pip. 

**If you are using uv:**

```bash
uv venv 
source .venv/bin/activate # Linux/macOS 

uv sync # to install dependencies
```

**If you are using pip:**

We also provide a requirements.txt file pip is preferred. Simply run:

```bash
pip install -r requirements.txt
```

## Data
We make use of the Sarcasm dataset provided by the [SocKet Repository](https://github.com/minjechoi/SOCKET). 

The dataset consists of claims with accompanying binary labels, indicating whether each claim is Literal or Sarcastic. 


## Code structure 
Main part of our experiments are carried out by the two python scripts: `run-eval.py` and `run-eval-r2.py`. They make use of the helper functions provided in `utils`. 
 
The `src` folder contains scripts for preprocessing of data, processing of inputs/outputs and formatting of results. 

The `configs` folder contains yaml files specifying models of interest, their parameters and the shared parameters across all models. 

The main results of our experiments are provided in `main-results.ipynb` and further plots and processing in `analysis.py`. 

### Experiments

In order to perform the experiment, follow the steps below. 

**Round $1$: Individual Inference**

You can run the first round experiment for a model by the following: 

```bash
uv run run-eval.py --model_name mistral-0.2-7b --outdir results/ --repetition 10 --dataset_path data/sarc/sarcasm.csv
```

You can specify any model from huggingface, given that the model's information is specified in the `models.yaml` file. 

**Curating data for Round $2$: Pairwise Interaction**

After Round $1$ is done for the models of interest, the data for Round $2$ needs to be created. 

In order to prepare the input for a given model of Round $2$, run the following: 

First we create a df that aggregate all model outputs on a model, claim level. 
```bash
uv run src/rate.py --round 1
```

We use this df to decide which claims and outputs to distribute to the model:
```bash
uv run src/prepare_round2.py mistral-0.2-7b
```

This will create a file, which will be the model's input in Round $2$. 

The input will consist of the claims, where the given agent disagrees highly with another agent. 
More specifically, over all $10$ repetitions, the model needs to predict a claim to be sarcastic less than three times, while another agent predicts the same claim to be sarcastic more than $7$ times. Or the other way around. 
The model is then prompted individually with all the other model's outputs for that instance. 

**Round $2$: Pairwise interaction**
You can run Round $2$ for a model with the following: 

```bash
uv run run-eval-round2.py --model_name mistral-0.3-7b --outdir /home/ --dataset_path /home/input_mistral-0.3-7b.csv
```
Make sure to specify the outdir and dataset path accordingly to your setup. 

**Formatting of results**

We aggregate the results from all Round $2$ runs. The results are on a modelA, modelB, claim level. 
```bash
uv run src/rate.py --round 2
```

Create main results file. This df is what the results are based on. These are further explored in the `main-results.ipynb` notebook. 
```bash
uv run src/results.py
```

## Overview of files and folders 
```
├── configs
│   ├── decoding.yaml
│   ├── default-model.yaml
│   └── models.yaml
├── data
│   └── sarc
│       ├── dummy.csv
│       ├── flagged.csv
│       ├── raw
│       │   ├── label_list.txt
│       │   ├── train_labels.txt
│       │   └── train_text.txt
│       └── sarcasm.csv
├── main-results.ipynb
├── plots
│   ├── heatmap-claim-count-r2-input.png
│   ├── heatmap-input-second.png
│   ├── heatmap-unique-claim-count-r2-input.png
│   ├── heatmaps-first-results.png
│   ├── label-dist-all.png
│   ├── label-distr-r1.png
│   └── valid-json-dist-r1.png
├── pyproject.toml
├── README.md
├── requirements.txt
├── results
│   ├── down_results.csv
│   ├── first-results-sarc-ratio.csv
│   ├── input-r2-claim-count.csv
│   ├── input-r2-claim-un_count.csv
│   ├── per_receiver_results.csv
│   ├── results.csv
│   ├── second-results-sarc-ratio.csv
│   └── up_results.csv
├── run-eval-round2.py
├── run-eval.py
├── src
│   ├── analysis.py
│   ├── influencing.py
│   ├── prepare_round2.py
│   ├── preprocessing.py
│   ├── ratio.py
│   └── results.py
├── utils
│   ├── __utils__.py
│   ├── data.py
│   ├── io.py
│   ├── models.py
│   ├── prompts.py
│   ├── runner.py
│   └── schemas.py
└── uv.lock
```