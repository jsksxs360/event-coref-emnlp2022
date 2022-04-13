# event-coref

pipline model for event coreference task on KBP dataset.

## Set up

#### Requirements

Set up a virtual environment and run: 

```bash
python3 -m pip install -r requirements.txt
```

#### Download the official evaluation script

Results of event coreference are obtained using ofﬁcial [**Reference Coreference Scorer**](https://github.com/conll/reference-coreference-scorers). This scorer reports results in terms of AVG-F, which is the unweighted average of the F-scores of four commonly used coreference evaluation metrics, namely **MUC** ([Vilain et al., 1995](https://www.aclweb.org/anthology/M95-1005/)), **B^3** ([Bagga and Baldwin, 1998](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.47.5848&rep=rep1&type=pdf)), **CEAFe** ([Luo, 2005](https://www.aclweb.org/anthology/H05-1004/)) and **BLANC** ([Recasens and Hovy, 2011](https://www.researchgate.net/profile/Eduard-Hovy/publication/231881781_BLANC_Implementing_the_Rand_index_for_coreference_evaluation/links/553122420cf2f2a588acdc95/BLANC-Implementing-the-Rand-index-for-coreference-evaluation.pdf)).

Run (from inside the repo):

```bash
git clone https://github.com/conll/reference-coreference-scorers.git
```

#### Prepare the dataset

This repo assumes access to the English corpora used in the TAC KBP Event Nugget Detection and Coreference task, i.e. [KBP 2015](http://cairo.lti.cs.cmu.edu/kbp/2015/event/), [KBP 2016](http://cairo.lti.cs.cmu.edu/kbp/2016/event/), and [KBP 2017]() corpus. In total, they contain 648 documents, which are either newswire articles or discussion forum threads. 

```
'2015': [
    'LDC_TAC_KBP/LDC2015E29/data/', 
    'LDC_TAC_KBP/LDC2015E68/data/', 
    'LDC_TAC_KBP/LDC2017E02/data/2015/training/', 
    'LDC_TAC_KBP/LDC2017E02/data/2015/eval/'
],
'2016': [
    'LDC_TAC_KBP/LDC2017E02/data/2016/eval/eng/nw/', 
    'LDC_TAC_KBP/LDC2017E02/data/2016/eval/eng/df/'
],
'2017': [
    'LDC_TAC_KBP/LDC2017E54/data/eng/nw/', 
    'LDC_TAC_KBP/LDC2017E54/data/eng/df/'
]
```

Following ([Lu & Ng, 2021 EMNLP](https://aclanthology.org/2021.emnlp-main.103/)), we select LDC2015E29, E68, E73, E94 and KBP 2016 test set as train & dev set (817 docs total, 735 for training and the remaining 82 for parameter tuning), and report results on the KBP 2017 test set.

Dataset Statistics:

|                  | Train | Dev  | Test |  All  |
| ---------------- | :---: | :--: | :--: | :---: |
| \#Documents      |  735  |  82  | 167  |  984  |
| \#Event mentions | 20512 | 2382 | 4375 | 27269 |
| \#Event Clusters | 13292 | 1502 | 2963 | 17757 |

1. Split sentences in source files using Stanford CoreNLP (see [readme](data/SplitSentences/readme.md)), create `kbp_sent.txt` in the *data* folder.

2. Convert the original dataset into jsonlines format using:

   ```
   cd data/
   export DATA_DIR=<ldc_tac_kbp_data_dir>
   python3 convert.py --kbp_data_dir $DATA_DIR
   ```

   Note: this script will create `train.json`、`dev.json` and `test.json` in the *data* folder, as well as `train_filtered.json`、`dev_filtered.json` and `test.json` which filter same and overlapping events.

## Training

