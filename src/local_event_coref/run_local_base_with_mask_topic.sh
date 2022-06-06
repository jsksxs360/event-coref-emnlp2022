export OUTPUT_DIR=./mask_topic_bert_results/

python3 run_local_base_with_mask_topic.py \
    --output_dir=$OUTPUT_DIR \
    --model_type=bert \
    --model_checkpoint=../../../PT_MODELS/bert-large-cased/ \
    --topic_model=vmf \
    --topic_dim=32 \
    --topic_inter_map=64 \
    --train_file=../../data/train_filtered.json \
    --dev_file=../../data/dev_filtered.json \
    --test_file=../../data/test_filtered.json \
    --max_seq_length=512 \
    --learning_rate=1e-5 \
    --matching_style=multi \
    --softmax_loss=ce \
    --num_train_epochs=10 \
    --batch_size=4 \
    --do_train \
    --warmup_proportion=0. \
    --seed=42