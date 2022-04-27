export OUTPUT_DIR=./longformer_roberta_results/
export CACHE_DIR=../../cache/

python3 run_global_base_with_local.py \
    --output_dir=$OUTPUT_DIR \
    --cache_dir=$CACHE_DIR \
    --model_type=longformer \
    --model_checkpoint=allenai/longformer-large-4096 \
    --mention_encoder_type=roberta \
    --mention_encoder_checkpoint=roberta-large \
    --train_file=../../data/train_filtered.json \
    --dev_file=../../data/dev_filtered.json \
    --test_file=../../data/test_filtered.json \
    --max_seq_length=4096 \
    --max_mention_length=512 \
    --learning_rate=1e-5 \
    --softmax_loss=ce \
    --num_train_epochs=50 \
    --batch_size=1 \
    --do_train \
    --do_test \
    --warmup_proportion=0. \
    --seed=42