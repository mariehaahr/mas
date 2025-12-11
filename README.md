# Multi-agent Systems 

## Project Overview



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

## Data


## Code structure 
Main part of our experiments are carried out by the two python scripts: `run-eval.py` and `run-eval-r2.py`. They make use of the helper functions provided in `utils`. 
 
The `src` folder contains scripts for preprocessing of data, processing of inputs/outputs and formatting of results. 

The configs folder contains yaml files specifying models of interest, their parameters and shared parameters across all models. 

The main results of our experiments are provided in `main-results.ipynb` and further plots and processing in `analysis.py`. 

### Experiments

In order to perform the experiment, follow the steps below. 

**First round: Solo Inference**

You can run the first round experiment for a model by the following: 

```bash
uv run run-eval.py --model_name mistral-0.2-7b --outdir results/ --repetition 10 --dataset_path data/sarc/sarcasm.csv
```

You can specify any model from huggingface, given that the model's information is specified in the `models.yaml` file. 

**Curating data for Second round: Pairwise Interaction**

After haven run all first round runs for the models of interest, the data for round 2 needs to be created. 

In order to prepare the input for a given model of round 2, run the following: 

First we create a df that aggregate all model outputs on a model, claim level. 
```bash
uv run src/rate.py --round 1
```

We use this df to decide which claims, outputs to distribute to which models:
```bash
uv run src/prepare_round2.py mistral-0.2-7b
```

This will create a file, which will be the model's input in round 2. 

The input will consists of the claims, where the given agent disagrees highly with another agent. 
More specifically, over all 10 repetitions, the model needs to predict a claim to be sarcastic less than three times, while another agent predicts the same claim to be sarcastic more than 7 times. Or the other way around. 
The model is then prompted individually with all the other model's outputs for that instance. 

**Second Round: Pairwise interaction**
You can run the second round of the experiments with the following: 

```bash
uv run run-eval-round2.py --model_name mistral-0.3-7b --outdir /home/ --dataset_path /home/input_mistral-0.3-7b.csv
```
Make sure to specify the outdir and dataset path accordingly to your wanted setup. 

**Various format of results**

We aggregate the results from all second run runs. The results are on a modelA, modelB, claim level. 
```bash
uv run src/rate.py --round 2
```

Create main results file. This df is what the majority of the results are based on. These are further explored in the `main-results.ipynb` notebook. 
```bash
uv run src/results.py
```






