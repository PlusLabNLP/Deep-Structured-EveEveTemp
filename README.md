# Basic Info
Author: Rujun Han, I-Hung Hsu, Mu Yang

Title: Codebase for CoNLL 2019 Paper: [Deep Structured Neural Network for Event
Temporal Relation Extraction](https://arxiv.org/pdf/1909.10094.pdf)

Data processinng. We have preprocessed MATRES(notice that the Matres dataset we
use are their initial released version, hence, contains less data), TB-Dense and 
TCR raw data using internal NLP tools at the Information Sciences Institute. 
These .pickle files are saved in data fold. 

# Setup

1. Install Gurobi
```
conda config --add channels http://conda.anaconda.org/gurobi
conda install gurobi
grbgetkey $YOURGUROBIKEY$
```

2. Return to this repository and:
```
pip install -r requirements.txt
```

3. Download data.

# Run code:
```
cd Code
python train_all.py
```
