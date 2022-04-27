export OUTPUT_DIR=./longformer_mask_results/
export CACHE_DIR=../../cache/

python3 run_pair_wise_coref_with_mask.py \
    --output_dir=$OUTPUT_DIR \
    --cache_dir=$CACHE_DIR \
    --model_type=longformer \
    --model_checkpoint=allenai/longformer-large-4096 \
    --train_file=../../data/train_filtered.json \
    --dev_file=../../data/dev_filtered.json \
    --test_file=../../data/test_filtered.json \
    --max_seq_length=4096 \
    --learning_rate=1e-5 \
    --softmax_loss=ce \
    --num_train_epochs=10 \
    --batch_size=4 \
    --do_train \
    --warmup_proportion=0. \
    --seed=42