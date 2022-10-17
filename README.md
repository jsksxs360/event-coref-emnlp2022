# Improving Event Coreference Resolution Using Document-level and Topic-level Information

This code was used in the paper:

**"Improving Event Coreference Resolution Using Document-level and Topic-level Information"**  
Sheng Xu, Peifeng Li and Qiaoming Zhu. EMNLP 2022.

A simple pipeline model implemented in PyTorch for resolving within-document event coreference. The model was trained and evaluated on the KBP corpus.

## Set up

#### Requirements

Set up a Python virtual environment and run: 

```bash
python3 -m pip install -r requirements.txt
```

#### Download the evaluation script

Coreference results are obtained using ofﬁcial [**Reference Coreference Scorer**](https://github.com/conll/reference-coreference-scorers). This scorer reports results in terms of AVG-F, which is the unweighted average of the F-scores of four commonly used coreference evaluation metrics, namely $\text{MUC}$ ([Vilain et al., 1995](https://www.aclweb.org/anthology/M95-1005/)), $B^3$ ([Bagga and Baldwin, 1998](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.47.5848&rep=rep1&type=pdf)), $\text{CEAF}_e$ ([Luo, 2005](https://www.aclweb.org/anthology/H05-1004/)) and $\text{BLANC}$ ([Recasens and Hovy, 2011](https://www.researchgate.net/profile/Eduard-Hovy/publication/231881781_BLANC_Implementing_the_Rand_index_for_coreference_evaluation/links/553122420cf2f2a588acdc95/BLANC-Implementing-the-Rand-index-for-coreference-evaluation.pdf)).

Run (from inside the repo):

```bash
cd ./
git clone https://github.com/conll/reference-coreference-scorers.git
```

#### Download pretrained models

Download the pretrained model weights (e.g. `bert-base-cased`) from Huggingface [Model Hub](https://huggingface.co/models):

```bash
bash download_pt_models.sh
```

**Note:** this script will download all pretrained models used in our experiment in `../PT_MODELS/`.

#### Prepare the dataset

This repo assumes access to the English corpora used in TAC KBP Event Nugget Detection and Coreference task (i.e., [KBP 2015](http://cairo.lti.cs.cmu.edu/kbp/2015/event/), [KBP 2016](http://cairo.lti.cs.cmu.edu/kbp/2016/event/), and [KBP 2017](http://cairo.lti.cs.cmu.edu/kbp/2017/event/)). In total, they contain 648 documents, which are either newswire articles or discussion forum threads. 

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

Following ([Lu & Ng, 2021](https://aclanthology.org/2021.emnlp-main.103/)), we select LDC2015E29, E68, E73, E94 and LDC2016E64 as train set (817 docs, 735 for training and the remaining 82 for parameter tuning), and report results on the KBP 2017 dataset.

**Dataset Statistics:**

|                  | Train | Dev  | Test |  All  |
| ---------------- | :---: | :--: | :--: | :---: |
| \#Documents      |  735  |  82  | 167  |  984  |
| \#Event mentions | 20512 | 2382 | 4375 | 27269 |
| \#Event Clusters | 13292 | 1502 | 2963 | 17757 |

Then, 

1. Split sentences and count verbs/entities in documents using Stanford CoreNLP (see [readme](data/SplitSentences/readme.md)), creating `kbp_sent.txt` and `kbp_word_count.txt` in the *data* folder.

2. Convert the original dataset into jsonlines format using:

   ```bash
   cd data/
   
   export DATA_DIR=<ldc_tac_kbp_data_dir>
   python3 convert.py --kbp_data_dir $DATA_DIR
   ```

   **Note:** this script will create `train.json`、`dev.json` and `test.json` in the *data* folder, as well as `train_filtered.json`、`dev_filtered.json` and `test_filtered.json` which filter same position and overlapping event mentions.

## Training

#### Trigger Detection

Train a sequence labeling model for Trigger Detection using the BIO tagging schema (Run with `--do_train`):

```bash
cd src/trigger_detection/

export OUTPUT_DIR=./softmax_ce_results/

python3 run_td_softmax.py \
    --output_dir=$OUTPUT_DIR \
    --model_type=longformer \
    --model_checkpoint=../../PT_MODELS/allenai/longformer-large-4096/ \
    --train_file=../../data/train_filtered.json \
    --dev_file=../../data/dev_filtered.json \
    --test_file=../../data/test_filtered.json \
    --max_seq_length=4096 \
    --learning_rate=1e-5 \
    --softmax_loss=ce \
    --num_train_epochs=50 \
    --batch_size=1 \
    --do_train \
    --warmup_proportion=0. \
    --seed=42
```

After training, the model weights and the evaluation results on **Dev** set would be saved in `$OUTPUT_DIR`.

#### Event Coreference

Train the full version of our event coreference model using (Run with `--do_train`):

```bash
cd src/global_event_coref/

export OUTPUT_DIR=./MaskTopic_M-multi-cosine_results/

python3 run_global_base_with_mask_topic.py \
    --output_dir=$OUTPUT_DIR \
    --model_type=longformer \
    --model_checkpoint=../../../PT_MODELS/allenai/longformer-large-4096/ \
    --mention_encoder_type=bert \
    --mention_encoder_checkpoint=../../../PT_MODELS/bert-base-cased/ \
    --topic_model=vmf \
    --topic_dim=32 \
    --topic_inter_map=64 \
    --train_file=../../data/train_filtered.json \
    --dev_file=../../data/dev_filtered.json \
    --test_file=../../data/test_filtered.json \
    --max_seq_length=4096 \
    --max_mention_length=256 \
    --learning_rate=1e-5 \
    --matching_style=multi_cosine \
    --softmax_loss=ce \
    --num_train_epochs=50 \
    --batch_size=1 \
    --do_train \
    --warmup_proportion=0. \
    --seed=42
```

After training, the model weights and evaluation results on **Dev** set would be saved in `$OUTPUT_DIR`.

## Evaluation

#### Trigger Detection

Run *run_td_softmax.py* with `--do_test`:

```bash
cd src/trigger_detection/

export OUTPUT_DIR=./softmax_ce_results/

python3 run_td_softmax.py \
    --output_dir=$OUTPUT_DIR \
    --model_type=longformer \
    --model_checkpoint=../../PT_MODELS/allenai/longformer-large-4096/ \
    --train_file=../../data/train_filtered.json \
    --dev_file=../../data/dev_filtered.json \
    --test_file=../../data/test_filtered.json \
    --max_seq_length=4096 \
    --learning_rate=1e-5 \
    --softmax_loss=ce \
    --num_train_epochs=50 \
    --batch_size=1 \
    --do_test \
    --warmup_proportion=0. \
    --seed=42
```

After evaluation, the evaluation results on **Test** set would be saved in `$OUTPUT_DIR`. Use `--do_predict` parameter to predict subtype labels. The predicted results, i.e., `XXX_test_pred_events.json`, would be saved in `$OUTPUT_DIR`. 

#### Event Coreference

Run *run_global_base_with_mask_topic.py* with `--do_test`:

```bash
cd src/global_event_coref/

export OUTPUT_DIR=./MaskTopic_M-multi-cosine_results/

python3 run_global_base_with_mask_topic.py \
    --output_dir=$OUTPUT_DIR \
    --model_type=longformer \
    --model_checkpoint=../../../PT_MODELS/allenai/longformer-large-4096/ \
    --mention_encoder_type=bert \
    --mention_encoder_checkpoint=../../../PT_MODELS/bert-base-cased/ \
    --topic_model=vmf \
    --topic_dim=32 \
    --topic_inter_map=64 \
    --train_file=../../data/train_filtered.json \
    --dev_file=../../data/dev_filtered.json \
    --test_file=../../data/test_filtered.json \
    --max_seq_length=4096 \
    --max_mention_length=256 \
    --learning_rate=1e-5 \
    --matching_style=multi_cosine \
    --softmax_loss=ce \
    --num_train_epochs=50 \
    --batch_size=1 \
    --do_test \
    --warmup_proportion=0. \
    --seed=42
```

After evaluation, the evaluation results on **Test** set would be saved in `$OUTPUT_DIR`. Use `--do_predict` parameter to predict coreferences for event mention pairs. The predicted results, i.e., `XXX_test_pred_corefs.json`, would be saved in `$OUTPUT_DIR`. 

#### Clustering

Create the final event clusters using predicted pairwise results:

```bash
cd src/clustering

export OUTPUT_DIR=./TEMP/

python3 run_cluster.py \
    --output_dir=$OUTPUT_DIR \
    --test_golden_filepath=../../data/test.json \
    --test_pred_filepath=../../data/XXX_weights.bin_test_pred_corefs.json \
    --golden_conll_filename=gold_test.conll \
    --pred_conll_filename=pred_test.conll \
    --do_evaluate
```

## Results

#### Trigger Detection

| Model                                                        | Micro (P / R / F1) | Macro (P / R / F1) |
| ------------------------------------------------------------ | :----------------: | :----------------: |
| [(Lu & Ng, 2021)](https://aclanthology.org/2021.emnlp-main.103/) | 71.6 / 58.7 / 64.5 |     - / - / -      |
| Longformer                                                   | 63.0 / 58.1 / 60.4 | 65.2 / 57.7 / 59.2 |
| Longformer+CRF                                               | 64.8 / 54.6 / 59.2 | 65.9 / 55.2 / 58.1 |

#### Classical Pairwise Models

| Model                       |      Pairwise      | MUC  |  B3  | CEA  | BLA  | AVG  |
| --------------------------- | :----------------: | :--: | :--: | :--: | :--: | :--: |
| BERT-large[Prod]            | 62.3 / 49.3 / 55.0 | 36.5 | 54.4 | 55.8 | 37.3 | 46.0 |
| RoBERTa-large[Prod]         | 64.6 / 44.0 / 52.4 | 36.0 | 54.8 | 55.6 | 37.3 | 45.9 |
| BERT-large[Prod] + Local    | 69.0 / 45.5 / 54.8 | 37.6 | 55.1 | 57.1 | 38.5 | 47.1 |
| RoBERTa-large[Prod] + Local | 71.7 / 49.9 / 58.9 | 39.0 | 55.8 | 58.0 | 39.6 | 48.1 |

#### Pairwise & Chunk Variants

Replace Global Mention Encoder in our model with pairwise (sentence-level) encoder or chunk (segment-level) encoder.

| Model                  |      Pairwise      | MUC  |  B3  | CEA  | BLA  | AVG  |
| ---------------------- | :----------------: | :--: | :--: | :--: | :--: | :--: |
| BERT-base[Pairwise]    | 64.0 / 39.8 / 49.0 | 35.3 | 54.4 | 55.8 | 36.6 | 45.5 |
| RoBERTa-base[Pairwise] | 59.9 / 55.6 / 57.7 | 39.0 | 54.3 | 56.4 | 38.6 | 47.1 |
| BERT-base[Chunk]       | 59.7 / 50.6 / 54.7 | 38.4 | 54.9 | 55.4 | 37.9 | 46.7 |
| RoBERTa-base[Chunk]    | 64.0 / 51.3 / 56.9 | 39.6 | 55.2 | 56.9 | 38.5 | 47.6 |

#### Our Model

| Model                                                        |      Pairwise      | MUC  |  B3  | CEA  | BLA  | AVG  |
| ------------------------------------------------------------ | :----------------: | :--: | :--: | :--: | :--: | :--: |
| [(Lu & Ng, 2021)](https://aclanthology.org/2021.emnlp-main.103/) |         -          | 45.2 | 54.7 | 53.8 | 38.2 | 48.0 |
| Global                                                       | 74.7 / 63.2 / 68.4 | 45.4 | 57.3 | 58.7 | 42.2 | 50.9 |
| + Local                                                      | 72.4 / 63.3 / 67.6 | 45.8 | 57.5 | 59.1 | 42.1 | 51.1 |
| + Local & Topic                                              | 72.0 / 64.4 / 68.0 | 46.2 | 57.4 | 59.0 | 42.0 | 51.2 |

#### Variants using different tensor matching

| Model              |      Pairwise      | MUC  |  B3  | CEA  | BLA  | AVG  |
| ------------------ | :----------------: | :--: | :--: | :--: | :--: | :--: |
| Base               | 37.5 / 48.0 / 42.1 | 36.7 | 54.9 | 55.3 | 34.7 | 45.4 |
| Base+Prod          | 71.2 / 64.0 / 67.4 | 45.4 | 57.0 | 58.6 | 41.2 | 50.5 |
| Base+Prod+Cos      | 72.0 / 64.4 / 68.0 | 46.2 | 57.4 | 59.0 | 42.0 | 51.2 |
| Base+Prod+Diff     | 70.3 / 67.1 / 68.7 | 45.0 | 56.7 | 58.9 | 41.4 | 50.5 |
| Base+Prod+Diff+Cos | 69.5 / 65.9 / 67.6 | 44.4 | 56.5 | 58.6 | 41.2 | 50.2 |

## Contact info

Contact [Sheng Xu](https://github.com/jsksxs360) at *[sxu@stu.suda.edu.cn](mailto:sxu@stu.suda.edu.cn)* for questions about this repository.

