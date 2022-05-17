export OUTPUT_DIR=./Mask_M-multi-cosine_closs_results/

python3 run_global_base_with_mask.py \
    --output_dir=$OUTPUT_DIR \
    --model_type=longformer \
    --model_checkpoint=../../../PT_MODELS/allenai/longformer-large-4096/ \
    --mention_encoder_type=bert \
    --mention_encoder_checkpoint=../../../PT_MODELS/bert-base-cased/ \
    --train_file=../../data/train_filtered.json \
    --dev_file=../../data/dev_filtered.json \
    --test_file=../../data/test_filtered.json \
    --max_seq_length=4096 \
    --max_mention_length=256 \
    --learning_rate=1e-5 \
    --add_contrastive_loss \
    --matching_style=multi_cosine \
    --softmax_loss=ce \
    --num_train_epochs=50 \
    --batch_size=1 \
    --do_train \
    --warmup_proportion=0. \
    --seed=42