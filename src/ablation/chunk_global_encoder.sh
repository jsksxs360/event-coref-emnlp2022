export OUTPUT_DIR=./ChunkBertEncoder_M-multi-cosine_results/

python3 chunk_global_encoder.py \
    --output_dir=$OUTPUT_DIR \
    --model_type=bert \
    --model_checkpoint=../../../PT_MODELS/bert-large-cased/ \
    --mention_encoder_type=bert \
    --mention_encoder_checkpoint=../../../PT_MODELS/bert-base-cased/ \
    --topic_model=vmf \
    --topic_dim=32 \
    --topic_inter_map=64 \
    --train_file=../../data/train_filtered.json \
    --dev_file=../../data/dev_filtered.json \
    --test_file=../../data/test_filtered.json \
    --max_seq_length=512 \
    --max_mention_length=256 \
    --learning_rate=1e-5 \
    --matching_style=multi_cosine \
    --softmax_loss=ce \
    --num_train_epochs=50 \
    --batch_size=1 \
    --do_train \
    --warmup_proportion=0. \
    --seed=42