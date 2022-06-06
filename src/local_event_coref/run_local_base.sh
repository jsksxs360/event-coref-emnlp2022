export OUTPUT_DIR=./longformer_results/

python3 run_local_base.py \
    --output_dir=$OUTPUT_DIR \
    --model_type=longformer \
    --model_checkpoint=../../../PT_MODELS/bert-large-cased/ \
    --train_file=../../data/train_filtered.json \
    --dev_file=../../data/dev_filtered.json \
    --test_file=../../data/test_filtered.json \
    --max_seq_length=512 \
    --learning_rate=1e-5 \
    --matching_style=multi_cosine \
    --softmax_loss=ce \
    --num_train_epochs=10 \
    --batch_size=4 \
    --do_train \
    --warmup_proportion=0. \
    --seed=42