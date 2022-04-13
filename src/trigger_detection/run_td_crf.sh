export OUTPUT_DIR=./crf_results/
export CACHE_DIR=../../cache/

python3 run_td_crf.py \
    --output_dir=$OUTPUT_DIR \
    --cache_dir=$CACHE_DIR \
    --model_checkpoint=allenai/longformer-large-4096 \
    --train_file=../../data/train_filtered.json \
    --dev_file=../../data/dev_filtered.json \
    --test_file=../../data/test_filtered.json \
    --max_seq_length=4096 \
    --learning_rate=1e-5 \
    --crf_learning_rate=5e-5 \
    --num_train_epochs=10 \
    --batch_size=1 \
    --do_train \
    --warmup_proportion=0. \
    --seed=42