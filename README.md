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

## Code structure 
 Main part of our experiments are carried out by the two python scripts: `run-eval.py` and `run-eval-r2.py`. They make use of the helper functions provided in `utils`. The `src` folder contains scripts for 

 The main results of our experiments are provided in `main-results.ipynb` and further plots and processing in `analysis.py`. 

### Reproducing results


## Data




