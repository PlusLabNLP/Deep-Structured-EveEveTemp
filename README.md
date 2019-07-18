# event-event-relations
Event-Event relations work under DARPA CwC

# Setup

1. Make a conda environment:
```
conda create -n cwc-event python=3.6
source activate cwc-event
```
2. Install FlexNlp (from outside this repository, usually a parent directory)
```
git clone https://github.com/isi-nlp/isi-flexnlp.git
cd isi-flexnlp
pip install -r requirements.txt
pip install -e .
python -m spacy download en
```
3. Install FlexNlp sandbox (from outside this repository, usually a parent directory):
```
git clone git@gitlab.com:isi-nlp/flexnlp-sandbox.git
cd flexnlp-sandbox
pip install -e .
```

3. Return to this repository and:
```
pip install -r requirements.txt
```

# Programs

* `red_to_vista_pickle` ingests RED annotation, converts it to Vista NLP format, and enriches it with spacy annotations
for feature extraction.



pip install -r requirements.txt
```

# Datasets

The datasets we use are available from:

* CaTeRS: http://cs.rochester.edu/nlp/rocstories/CaTeRS/
* RED: https://catalog.ldc.upenn.edu/LDC2016T23
