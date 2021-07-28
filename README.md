# Basic Info
Author: Rujun Han*, I-Hung Hsu*, Mu Yang

Title: Codebase for CoNLL 2019 Paper: [Deep Structured Neural Network for Event
Temporal Relation Extraction](https://arxiv.org/pdf/1909.10094.pdf)

Data processinng. We have preprocessed MATRES(notice that the Matres dataset we
use are their initial released version, hence, contains less data), TB-Dense and 
TCR raw data using internal NLP tools at the Information Sciences Institute. 
These .pickle files are saved in data fold. 

## Additional Note:
- If you are curious about the data preprocessing, we recommend you to see this
  script : https://github.com/rujunhan/TEDataProcessing/blob/master/processMATRES.py
- However, this is not been test in this repo, thus, the best case is to
  download data from the link we provide

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

Or see the yml file we append

3. Download data.

# Run code:
```
cd Code
python train_all.py
```
