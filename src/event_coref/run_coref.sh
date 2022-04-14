export OUTPUT_DIR=./results/
export CACHE_DIR=../../cache/

python3 run_coref.py \
    --output_dir=$OUTPUT_DIR \
    --cache_dir=$CACHE_DIR \
    --model_checkpoint=allenai/longformer-large-4096 \
    --train_file=../../data/train_filtered.json \
    --dev_file=../../data/dev_filtered.json \
    --test_file=../../data/test_filtered.json \
    --max_seq_length=4096 \
    --learning_rate=1e-5 \
    --softmax_loss=ce \
    --num_train_epochs=50 \
    --batch_size=1 \
    --do_train \
    --do_test \
    --warmup_proportion=0. \
    --seed=42